#!/usr/bin/env python3
"""録音を Gemini に文字起こしさせて、こちらの transcripts.jsonl と同じ形で書き出す。

ローカルの mlx-whisper と精度を比べるための道具（2026-09-12 運用者 依頼「日本語の文字起こしが
優秀と聞いた。測ってみよう」）。出力は `scripts/score_transcript.py` でそのまま採点できる。

    python scripts/gemini_transcribe.py 録音.wav --out out.jsonl --speaker 自分
    python scripts/gemini_transcribe.py 相手.wav --out out.jsonl --diarize   # 話者分けも任せる

API キーは環境変数 GEMINI_API_KEY か `--key-file` から読む（画面にもログにも出さない）。
  チャットや履歴に残さないため、キーはファイルに置いて渡す（例: ~/.config/meeting-copilot/gemini.key）。

入口が 2 つある。モデル名で切り替わる（2026-09-13 に実測で判明）。

  1. `gemini-3.5-transcribe`（既定）= 文字起こし専用。
     **`generateContent` では動かない。** 音声だけ・指示つき・mp3・wav 16k・streaming の 5 通り
     すべてで finishReason=STOP のまま本文が空（音声トークンは 1,500/分と数えられるので、
     届いていないのではなく何も返していない）。`responseSchema` を付けると
     `JSON mode is not enabled for this model` の 400。
     正しい入口は **`POST /v1beta/interactions`**、音声は **Files API に上げて uri で渡す**
     （インライン base64 は不可）。返るのは語ごとの `word_info`（話者・開始・終了）。
     ・話者分け／語の時刻を付けると **音声は 30 分まで**（付けなければ 1 時間）
     ・`mode: smart` は時刻・話者と併用できない（相づちや言い直しを整理する代わり）
     ・用語リストは 1,000 語まで渡せるが、**時刻とは併用できない**
       （`custom_vocabulary is incompatible with timestamps.` の 400）。
       こちらは認識の**後**に config/glossary.yaml で置換しているので、時刻を取るほうを選ぶ。
       用語リストを使いたいときは `--no-timestamps` を付ける（そのとき時刻は行の順番だけ）

  2. `gemini-3.5-flash` など汎用モデル = `generateContent` + `responseSchema`。
     同じ音声をきちんと起こすが、**10 分で投げると 7.9 分あたりで勝手に打ち切られる**
     （区間の 21% が欠けた）。だから汎用側の刻みは 5 分。
     思考トークンは切る（`thinkingBudget: 0`）。入れたままだと本文 621 に対し思考 2,332＝
     出力の費用が 4 倍になる。文字起こしに推論は要らない。`thinkingLevel: "off"` は 400。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stt.gemini_transcribe import (  # noqa: E402
    BASE,
    TranscribeApi,
    TranscribeConfig,
    post as _post,
    with_retry as _with_retry,
)

ENDPOINT = BASE + "/v1beta/models/{model}:generateContent"

PROMPT_PLAIN = """この音声を文字起こししてください。日本語の会議です。

- 発話ごとに 1 件、開始・終了の秒数（この音声の先頭を 0 とする）と本文を出す
- 聞き取れない部分は無理に補わない。相づち（はい・うん）も 1 件として出す
- 要約・言い換えはしない。話された言葉をそのまま書く
"""

PROMPT_DIARIZE = PROMPT_PLAIN + """- 話者が複数いる。声で聞き分けて speaker に「話者1」「話者2」…を入れる
  （名前が音声中で分かる場合だけ、その名前を使ってよい）
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        }
    },
    "required": ["utterances"],
}


def to_mp3(source: Path, start: float, duration: float, out_path: Path) -> None:
    """指定区間を mp3（モノラル 64kbps）に切り出す。トークンは長さで決まるので音質は上げない。"""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(duration),
         "-i", str(source), "-ac", "1", "-b:a", "64k", str(out_path)],
        check=True,
    )


def audio_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def require_paid_project(project: str) -> None:
    """課金が有効なプロジェクトでなければ、音声を 1 バイトも送らずに止める。

    なぜ要るか（2026-09-13 の事故）: Gemini API の**無料枠は、送った内容を製品改善に使い、
      人間のレビュアーが読むことがある**（規約に「機密情報を送るな」と明記）。有料枠では使われない。
      この違いを確認せずにクライアント会議 50 分（金額 13 か所・実名 48 か所）を無料枠へ送った。
      アップロードした音声は消せるが、**改善用のログは自分では消せない**。
    ∴ 送る前に課金を機械的に確かめる。判定できないときも止める（fail-closed）。
    """
    result = subprocess.run(
        ["gcloud", "billing", "projects", "describe", project, "--format", "value(billingEnabled)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{project} の課金状態を確認できませんでした（gcloud の認証が切れている？）。\n"
            f"  {result.stderr.strip()[:200]}\n"
            "確認できないので送信しません。`gcloud auth login` のあとやり直してください。"
        )
    if result.stdout.strip() != "True":
        raise SystemExit(
            f"{project} は課金が有効ではありません。無料枠は送った内容が製品改善に使われ、\n"
            "人間のレビュアーが読むことがあります（規約が機密情報の送信を禁じています）。\n"
            "会議音声は送信しません。"
        )


def mean_volume_dbfs(path: Path, start: float, duration: float) -> float:
    """指定区間の平均音量（dBFS）を返す。完全な無音は -91 付近になる。"""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-ss", str(start), "-t", str(duration), "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    found = re.search(r"mean_volume:\s*(-?[\d.]+)", result.stderr)
    return float(found.group(1)) if found else 0.0


def content_end(path: Path, total: float, *, window: float = 60.0, floor_dbfs: float = -80.0) -> float:
    """末尾の無音を除いた「中身のある終わり」の秒数を返す。

    なぜ要るか（2026-09-13 実測）: 録音は会議より長い。09-11 の録音は 67.2 分あるが、
      音があるのは 53 分まで（以降は -91 dBFS ＝ デジタル無音）。無音混じりの 17 分を
      まとめて投げると、専用モデルは **何も返さない**（入力 25,731 トークンは数えられるのに
      0 発話）。末尾を落としておけば、最後の区間も中身だけになって通る。
    """
    position = total
    while position > window:
        if mean_volume_dbfs(path, position - window, window) > floor_dbfs:
            return position
        position -= window
    return position


def load_vocabulary(path: Path, limit: int = 1000) -> list[str]:
    """辞書から「正しい表記」だけを取り出して用語リストにする。

    置き換え辞書は `誤り: 正しい` なので、渡すのは値のほう。Whisper のヒントと違って
      これはモデルの語彙に足すだけなので、文中に紛れ込む心配がない（2026-09-11 の失敗とは別物）。
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    terms: list[str] = []
    for value in (data.get("replacements") or {}).values():
        term = str(value).strip()
        if term and term not in terms:
            terms.append(term)
    return terms[:limit]


def ask_transcribe(model: str, path: Path, *, language: str, vocabulary: list[str],
                   diarize: bool, timestamps: bool, timeout: float) -> dict:
    """専用モデルに 1 区間を投げる。返りは汎用側と同じ {"data": {...}, "usage": {...}}。

    口の実装は `src/stt/gemini_transcribe.py`（会議中の 30 秒刻みと同じものを使う）。
    """
    api = TranscribeApi(os.environ["GEMINI_API_KEY"], TranscribeConfig(
        model=model, language=language, timeout_sec=timeout,
        diarize=diarize, timestamps=timestamps, vocabulary=vocabulary))
    result = api.transcribe_file(path)
    return {"data": {"utterances": result["utterances"]}, "usage": result["usage"]}


# ---------------------------------------------------------------------------
# 入口 2: 汎用モデル（generateContent + responseSchema）
# ---------------------------------------------------------------------------

def ask_gemini(model: str, audio: bytes, prompt: str, timeout: float) -> dict:
    """音声 1 区間を投げて JSON を受け取る。429 は待って繰り返す。"""
    body = {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/mp3", "data": base64.b64encode(audio).decode()}},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},   # 文字起こしに推論は要らない。冒頭の注記参照
        },
    }
    _, raw = _with_retry(lambda: _post(
        ENDPOINT.format(model=model), json.dumps(body).encode(),
        {"content-type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        timeout))
    payload = json.loads(raw)
    candidate = (payload.get("candidates") or [{}])[0]
    text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []))
    usage = payload.get("usageMetadata", {})
    if not text.strip():
        raise RuntimeError(f"空の応答（{candidate.get('finishReason')}）")
    # 思考トークンも出力として課金される。切ったつもりで残っていたら気づけるように足す
    usage = dict(usage)
    usage["candidatesTokenCount"] = (usage.get("candidatesTokenCount", 0)
                                     + usage.get("thoughtsTokenCount", 0))
    return {"data": json.loads(text), "usage": usage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-transcribe",
                        help="既定は文字起こし専用モデル。汎用と比べるなら gemini-3.5-flash")
    parser.add_argument("--chunk-min", type=float,
                        help="1 回に投げる長さ（分）。既定は専用 25 分／汎用 5 分")
    parser.add_argument("--speaker", default="話者", help="固定の話者名（自分側の録音に使う）")
    parser.add_argument("--diarize", action="store_true", help="話者分けも Gemini に任せる")
    parser.add_argument("--language", default="ja-JP")
    parser.add_argument("--vocab", type=Path, help="用語リストに使う置き換え辞書（専用モデルのみ）")
    parser.add_argument("--no-timestamps", action="store_true",
                        help="語の時刻を求めない。これを付けたときだけ --vocab が効く")
    parser.add_argument("--offset", type=float, default=0.0, help="出力の時刻に足す秒数")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--keep-silence", action="store_true",
                        help="末尾の無音を落とさない（既定は落とす）")
    parser.add_argument("--key-file", type=Path, help="API キーを書いたファイル（1 行目だけ読む。中身は出力しない）")
    parser.add_argument("--billing-project", default=os.environ.get("GEMINI_BILLING_PROJECT"),
                        help="キーの所属プロジェクト ID。課金が有効か確かめてから送る（必須）")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.key_file:
        key = args.key_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        if not key:
            raise SystemExit(f"{args.key_file} が空です")
        os.environ["GEMINI_API_KEY"] = key      # 値は表示しない
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY が設定されていません（--key-file でも渡せます）")

    # 音声を外へ出す前の関門。冒頭の require_paid_project の注記を読むこと
    if not args.billing_project:
        raise SystemExit(
            "--billing-project（または環境変数 GEMINI_BILLING_PROJECT）が要ります。\n"
            "キーの所属プロジェクト ID を渡してください。課金が有効かを確かめてから送ります。\n"
            "無料枠は送った内容が製品改善に使われ、人間のレビュアーが読むことがあります。"
        )
    require_paid_project(args.billing_project)

    dedicated = "transcribe" in args.model
    # 専用モデルは話者分け・語の時刻を付けると 30 分まで。余裕を見て 25 分で刻む
    chunk_min = args.chunk_min if args.chunk_min else (25.0 if dedicated else 5.0)
    vocabulary = load_vocabulary(args.vocab) if (args.vocab and dedicated) else []
    if args.vocab and not dedicated:
        print("※ 用語リストは専用モデルだけの機能なので無視します", file=sys.stderr)
    elif vocabulary and not args.no_timestamps:
        print("※ 用語リストは時刻と併用できないので今回は使いません（--no-timestamps で使えます）",
              file=sys.stderr)
        vocabulary = []

    recorded = audio_seconds(args.audio)
    total = recorded if args.keep_silence else content_end(args.audio, recorded)
    chunk = chunk_min * 60
    count = int(total // chunk) + (1 if total % chunk else 0)
    extra = f" / 用語 {len(vocabulary)} 語" if vocabulary else ""
    dropped = (f"（末尾の無音 {(recorded - total) / 60:.1f} 分を落とした）"
               if recorded - total >= 30 else "")
    print(f"{args.audio.name}: 録音 {recorded / 60:.1f} 分 → 中身 {total / 60:.1f} 分{dropped}"
          f" → {count} 回に分けて {args.model} へ（1 回 {chunk_min:g} 分{extra}）", flush=True)

    rows: list[dict] = []
    tokens = {"prompt": 0, "output": 0}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(count):
            if args.limit_chunks is not None and index >= args.limit_chunks:
                break
            begin = index * chunk
            piece = Path(tmp) / f"part{index}.mp3"
            to_mp3(args.audio, begin, min(chunk, total - begin), piece)
            try:
                if dedicated:
                    result = ask_transcribe(args.model, piece, language=args.language,
                                            vocabulary=vocabulary, diarize=args.diarize,
                                            timestamps=not args.no_timestamps,
                                            timeout=args.timeout)
                else:
                    prompt = PROMPT_DIARIZE if args.diarize else PROMPT_PLAIN
                    result = ask_gemini(args.model, piece.read_bytes(), prompt, args.timeout)
            except (urllib.error.URLError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
                print(f"  [{index + 1}/{count}] 失敗: {exc}", file=sys.stderr, flush=True)
                continue
            usage = result["usage"]
            tokens["prompt"] += usage.get("promptTokenCount", 0)
            tokens["output"] += usage.get("candidatesTokenCount", 0)
            found = result["data"].get("utterances", [])
            for item in found:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                rows.append({
                    "speaker": str(item.get("speaker") or args.speaker) if args.diarize else args.speaker,
                    "text": text,
                    "start_time": round(begin + float(item.get("start", 0.0)) + args.offset, 2),
                    "end_time": round(begin + float(item.get("end", 0.0)) + args.offset, 2),
                    "timestamp": "",
                })
            last = max((row["end_time"] for row in rows), default=0.0)
            print(f"  [{index + 1}/{count}] {len(found)} 発話 / {last:.0f} 秒目まで"
                  f"（入力 {usage.get('promptTokenCount', 0)} ・"
                  f"出力 {usage.get('candidatesTokenCount', 0)} トークン）", flush=True)

    rows.sort(key=lambda row: row["start_time"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    # 料金は 2026-09 時点の Gemini 3 Flash 系（音声入力 $1/1M・出力 $3/1M）で概算。
    #   専用モデルの単価は未確認なので、これは「汎用で同じ量を投げたら」の目安として読む
    cost = tokens["prompt"] / 1e6 * 1.0 + tokens["output"] / 1e6 * 3.0
    print(f"\n{len(rows)} 発話 / {elapsed:.0f} 秒 / 入力 {tokens['prompt']} ・出力 {tokens['output']} トークン"
          f"（概算 ${cost:.2f}）\n書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
