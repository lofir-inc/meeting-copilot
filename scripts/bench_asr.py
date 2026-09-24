#!/usr/bin/env python3
"""文字起こしのエンジンを並べて比べる（日本語でどれが強いか）。

    # 手元だけ（外へ何も出さない）
    ./scripts/bench_asr.py --audio sample.wav --reference sample.txt --engines local

    # 外のエンジンも入れる（公開データ限定。実会議の音声は拒む）
    ./scripts/bench_asr.py --audio sample.wav --reference sample.txt \\
        --engines local,gemini,deepgram --allow-external

出すもの: エンジンごとの **文字の食い違い率 / 所要時間 / 実時間に対する速さ / 費用の目安**。

なぜ要るか（2026-09-18）: 「Deepgram は Gemini より日本語が強いか」を**記憶で答えない**ため。
  日本語 ASR の優劣は、英語の評判とは別に測らないと分からない。ここに並べれば同じ物差しで比べられる。

**実会議の音声を外へ出さない。** `workspace/sessions/` の下の音声を外のエンジンへ渡そうとすると
  止まる（CLAUDE.md の関門: 外部 API へ音声を出すには 運用者 の事前承認と、課金が有効な
  プロジェクトの機械的な確認が要る）。比較は**公開データ**で行い、勝った 1 社だけを、
  承認を得たうえで実会議 1 本にかける。

新しいエンジンを足すのは `ENGINES` に 1 つ関数を書くだけ。キーは
  `~/.config/meeting-copilot/<名前>.key` の 1 行目を読む（Gemini だけは `settings.yaml` の場所）。

**作り物の音声（`say`）では差が出ない。** 2026-09-18 の実測:

    local  7.1%  1.8 秒（18.2 倍速）   gemini 10.0%  9.4 秒（3.5 倍速）

  どちらも文としては完全で、差は「2万ミリメートル / 2万mm」のような表記ゆれだけだった。
  **エンジンの優劣は、雑音・重なり・言い淀みのある会話でしか出ない。**
  比べるなら**日本語の会話**の公開データを使うこと（朗読データでも不足）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.score_transcript import Row, edit_distance  # noqa: E402

KEY_DIR = Path("~/.config/meeting-copilot").expanduser()
PRICES = {                      # 1 音声時間あたりの目安（円・要確認）
    "local": 0.0,
    "gemini": 83.0,             # 2026-09-17 の実測（文字起こし＋要約の合計 ¥1.39/分）のうち音声ぶん
    "deepgram": 39.0,           # 公表価格からの換算（未検証）
}


def read_key(name: str) -> str:
    """キーの 1 行目。中身は表示しない。場所は設定を先に見る（決め打ちしない）。"""
    path = _key_path(name)
    if not path.is_file():
        raise SystemExit(f"キーがありません: {path}（1 行目に書いてください・chmod 600）")
    return path.read_text(encoding="utf-8").splitlines()[0].strip()


def _key_path(name: str) -> Path:
    """Gemini は `settings.yaml` の場所を使う。ほかは `~/.config/meeting-copilot/<名前>.key`。"""
    if name == "gemini":
        import yaml

        settings_path = REPO / "config" / "settings.yaml"
        if settings_path.exists():
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
            told = (((settings.get("meeting") or {}).get("external_stt") or {}).get("key_file") or "")
            if told:
                return Path(str(told)).expanduser()
    return KEY_DIR / f"{name}.key"


def refuse_real_meetings(audio: Path, engine: str) -> None:
    """実会議の音声を外のエンジンへ渡さない（関門）。"""
    if engine == "local":
        return
    resolved = audio.resolve()
    if (REPO / "workspace" / "sessions").resolve() in resolved.parents:
        raise SystemExit(
            f"実会議の音声は外へ出せません: {resolved}\n"
            "  比較は公開データで行ってください（勝ったエンジンだけ、承認を得て実会議にかける）。")


# ------------------------------------------------------------------ エンジン

def run_local(audio: Path) -> str:
    """手元の Whisper（mlx-whisper）。外へ出ない。

    **まるごと 1 回で渡す**。Whisper は自分で 30 秒ずつ刻むので、こちらで切ると損をする
    （2026-09-18 の実測: まるごと 13.0% ／ 25 秒でぶつ切り 22.6% ／ 静かなところで切る 29.9%）。
    会議用の設定は「いちばん自信の低い行」で全部捨てるので、比較では外す
    （`min_avg_logprob=None`）。長い音声だと 1 か所の落ち込みで**何も返らない**。
    """
    import mlx_whisper
    import numpy as np
    import soundfile as sf

    from src.stt.whisper_client import SttConfig, WhisperClient

    samples, rate = sf.read(str(audio), dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = np.mean(samples, axis=1)
    config = SttConfig(min_avg_logprob=None)
    WhisperClient(config)._ensure_model()
    result = mlx_whisper.transcribe(samples, path_or_hf_repo=config.model,
                                    language=config.language, condition_on_previous_text=False)
    return str(result.get("text", ""))


def run_gemini(audio: Path) -> str:
    """Gemini の文字起こし（音声を外へ出す）。"""
    from src.stt.gemini_transcribe import TranscribeApi, TranscribeConfig

    api = TranscribeApi(read_key("gemini"), TranscribeConfig())
    result = api.transcribe_file(audio)
    return "".join(str(one.get("text", "")) for one in (result.get("utterances") or []))


def run_deepgram(audio: Path) -> str:
    """Deepgram（音声を外へ出す）。日本語は未検証 — ここで測るのが目的。"""
    import httpx

    with httpx.Client(timeout=300) as client:
        response = client.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": "nova-3", "language": "ja", "smart_format": "true"},
            headers={"Authorization": f"Token {read_key('deepgram')}",
                     "Content-Type": "audio/wav"},
            content=audio.read_bytes())
    response.raise_for_status()
    body = response.json()
    parts = body["results"]["channels"][0]["alternatives"]
    return parts[0]["transcript"] if parts else ""


ENGINES = {"local": run_local, "gemini": run_gemini, "deepgram": run_deepgram}


# ------------------------------------------------------------------ 採点

KANJI_DIGITS = {"〇": "0", "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
                "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
UNITS = {"ミリメートル": "mm", "センチメートル": "cm", "メートル": "m", "キロメートル": "km",
         "パーセント": "%", "キログラム": "kg", "グラム": "g", "円": "円"}


def canonical(text: str) -> str:
    """数字と単位の書き方をそろえる。

    2026-09-18 に踏んだ: 作り物の音声で Deepgram 1.2% / Whisper 7.1% / Gemini 10.0% と出たが、
    中身を見ると**差は「三つ / 3つ」「二万ミリメートル / 2万mm」だけ**だった。
    読み上げ原稿は漢数字なので、漢数字のまま出すエンジンが有利になる＝**精度ではなく表記の一致**。
    数字と単位をそろえてから比べる。
    """
    # 順番が大事: 記号を落とす前に単位を置き換える（`_STRIP` が長音「ー」を消すので、
    #   先に落とすと「ミリメートル」が「ミリメトル」になって当たらない。2026-09-18 に踏んだ）
    canon = unicodedata.normalize("NFKC", text).lower()
    for word, mark in UNITS.items():
        canon = canon.replace(word.lower(), mark)
    for kanji, digit in KANJI_DIGITS.items():
        canon = canon.replace(kanji, digit)
    canon = canon.replace("十", "").replace("百", "").replace("千", "")
    canon = canon.replace("万", "0000").replace("億", "00000000")
    return Row(0.0, 0.0, "", canon).normalized


def compare(hypothesis: str, reference: str) -> dict:
    """文字の食い違い率（`score_transcript.py` と同じ距離。数字と単位はそろえてから比べる）。"""
    ours, theirs = canonical(hypothesis), canonical(reference)
    distance = edit_distance(ours, theirs)
    return {"chars": len(theirs), "distance": distance,
            "error_rate": round(distance / max(len(theirs), 1), 4)}


def bench(audio: Path, reference: str, engines: list[str], *, allow_external: bool) -> list[dict]:
    rows = []
    seconds = audio_seconds(audio)
    for name in engines:
        run = ENGINES.get(name)
        if run is None:
            raise SystemExit(f"知らないエンジンです: {name}（{'/'.join(ENGINES)}）")
        refuse_real_meetings(audio, name)
        if name != "local" and not allow_external:
            raise SystemExit(f"{name} は音声を外へ出します。公開データであることを確かめて "
                             "--allow-external を付けてください。")
        started = time.time()
        try:
            text = run(audio)
            error = ""
        except Exception as exc:                     # noqa: BLE001 - 1 つ落ちても他を測る
            text, error = "", f"{type(exc).__name__}: {exc}"[:120]
        took = time.time() - started
        row = {"engine": name, "seconds": round(took, 1),
               "speed": round(seconds / took, 1) if took and seconds else 0,
               "yen_per_hour": PRICES.get(name, 0), "error": error, "text": text}
        row.update(compare(text, reference) if text else {"error_rate": 1.0})
        rows.append(row)
    return rows


def audio_seconds(audio: Path) -> float:
    import wave

    try:
        with wave.open(str(audio), "rb") as handle:
            return handle.getnframes() / handle.getframerate()
    except Exception:                                # noqa: BLE001 - wav 以外なら長さは出さない
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="正解のテキスト（読み上げた原稿など）")
    parser.add_argument("--engines", default="local")
    parser.add_argument("--allow-external", action="store_true", help="音声を外へ出すエンジンを許す")
    parser.add_argument("--out", type=Path, help="結果を JSON で残す")
    args = parser.parse_args()

    reference = args.reference.read_text(encoding="utf-8")
    rows = bench(args.audio, reference, [name.strip() for name in args.engines.split(",") if name.strip()],
                 allow_external=args.allow_external)

    length = audio_seconds(args.audio)
    print(f"\n音声: {args.audio.name}（{length:.0f} 秒）／正解 {len(reference)} 文字\n")
    print(f"  {'エンジン':<10}{'食い違い':>8}{'所要':>8}{'速さ':>8}   費用の目安")
    for row in sorted(rows, key=lambda one: one["error_rate"]):
        if row["error"]:
            print(f"  {row['engine']:<10}{'—':>8}{'—':>8}{'—':>8}   {row['error']}")
            continue
        print(f"  {row['engine']:<10}{row['error_rate']:>7.1%}{row['seconds']:>7.1f}s"
              f"{row['speed']:>7.1f}x   ¥{row['yen_per_hour']:.0f}/時間")
    print("\n食い違い率は「正解との差」であって正解率ではありません（正解も人が作った 1 つの版です）。")
    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"　残しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
