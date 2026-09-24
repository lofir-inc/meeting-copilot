#!/usr/bin/env python3
"""録音を `gemini-3.5-transcribe-live` へ流して、会議中の口の実力を測る。

    ./scripts/gemini_live_transcribe.py 録音.wav --out out.jsonl --billing-project <p>

会議中に使う経路（WebSocket・`bidiGenerateContent`）を、録音を流し込むことで再現する。
出力は `scripts/score_transcript.py` でそのまま採点できる。

口の形（2026-09-13 に実測で確定。**推測で作らない**）
  - 入口は `wss://…/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=`
  - setup は `{"setup": {"model": "models/…", "input_audio_transcription": {}}}`
    HTTP 側の `generation_config.transcription_config` は **400 で弾かれる**（別物）
  - 返ってくるもの:
      `voiceActivity` … `ACTIVITY_START` / `ACTIVITY_END` と `audioOffset`（音声先頭からの秒）
      `serverContent.interimInputTranscription` … 途中経過
      `serverContent.inputTranscription`        … 確定テキスト（発話ごと）
  - 確定は発話が終わってから **0.16〜0.41 秒**で返る（実時間で流したとき）
  - **セッションには時間の上限がある**（実測: 約 10 分で `goAway` → 1008 で切断）。
    50 分の会議は 1 本では通らない。予告が来たら送るのをやめ、繋ぎ直して続きから流す。
    `audioOffset` は接続ごとに 0 から数え直されるので、開始位置を足して全体の時刻に直す。
  話者は付かない。VAD の時刻で手元の録音を切って resemblyzer に渡す
    （`src/audio/segment_labeler.py`。2026-09-11 の実会議で 97%）

送る前に `--billing-project` の課金を確かめる（無料枠は送った内容が製品改善に使われる）。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.stt.external_consent import billing_enabled  # noqa: E402

HOST = "generativelanguage.googleapis.com"
PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
SEND_RATE = 16000
FRAME_SEC = 0.2
DRAIN_SEC = 12.0
"""goAway のあと、処理中の確定を待つ上限。"""


def load_pcm16(path: Path, rate: int) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg に失敗: {result.stderr.decode()[:200]}")
    return result.stdout


def offset_seconds(value: object) -> float:
    text = str(value or "0").rstrip("s")
    try:
        return float(text)
    except ValueError:
        return 0.0


async def one_session(model: str, key: str, pcm: bytes, base_sec: float, speed: float,
                      speaker: str, rows: list[dict], latencies: list[float]) -> int:
    """1 本の接続で流せるところまで流し、**送れたバイト数**を返す。

    セッションには時間の上限がある（2026-09-13 実測: 約 10 分で GoAway → 切断）。
      50 分の会議は 1 本では通らないので、呼び出し側が繋ぎ直して続きを流す。
    `audioOffset` は**接続ごとに 0 から**数え直されるので、`base_sec` を足して
      録音全体の時刻に直す。
    """
    import websockets

    frame = int(FRAME_SEC * SEND_RATE) * 2
    pending_start = 0.0
    sent = 0
    going_away = asyncio.Event()
    last_done = [base_sec]          # 確定が取れた最後の時刻（全体の秒）

    async with websockets.connect(f"wss://{HOST}{PATH}?key={key}", max_size=None,
                                  open_timeout=60, ping_interval=20) as socket:
        await socket.send(json.dumps({"setup": {"model": f"models/{model}",
                                                "input_audio_transcription": {}}}))
        await socket.recv()          # setupComplete
        started = time.perf_counter()

        async def listen() -> None:
            nonlocal pending_start
            try:
                async for raw in socket:
                    message = json.loads(raw)
                    if "goAway" in message:
                        # 予告が来たら送るのをやめる。無視して送り続けると 1008 で切られる
                        going_away.set()
                        continue
                    content = message.get("serverContent") or {}
                    activity = message.get("voiceActivity")
                    if activity and activity.get("type") == "ACTIVITY_START":
                        pending_start = base_sec + offset_seconds(activity.get("audioOffset"))
                    if "inputTranscription" in content:
                        text = (content["inputTranscription"] or {}).get("text", "").strip()
                        if text:
                            rows.append({"speaker": speaker, "text": text,
                                         "start_time": round(pending_start, 2),
                                         "end_time": round(pending_start, 2), "timestamp": ""})
                    if activity and activity.get("type") == "ACTIVITY_END":
                        end = base_sec + offset_seconds(activity.get("audioOffset"))
                        if rows and rows[-1]["end_time"] <= rows[-1]["start_time"]:
                            rows[-1]["end_time"] = round(end, 2)
                        latencies.append(time.perf_counter() - started
                                         - (end - base_sec) / max(speed, 0.001))
                        last_done[0] = end
            except Exception:        # noqa: BLE001 — 接続が閉じたら抜ける
                going_away.set()

        listener = asyncio.create_task(listen())
        try:
            for index in range(0, len(pcm), frame):
                if going_away.is_set():
                    break
                await socket.send(json.dumps({"realtime_input": {"audio": {
                    "data": base64.b64encode(pcm[index:index + frame]).decode(),
                    "mime_type": f"audio/pcm;rate={SEND_RATE}"}}}))
                sent = min(index + frame, len(pcm))
                if speed > 0:
                    await asyncio.sleep(FRAME_SEC / speed)
            else:
                await socket.send(json.dumps({"realtime_input": {"audio_stream_end": True}}))
                await asyncio.sleep(6)
            # goAway で抜けたときも、処理中の確定が返るのを待つ。
            #   打ち切ると送信ずみの音声ぶんが丸ごと消える（2026-09-13 実測: 繋ぎ目で 17〜35 秒）
            for _ in range(int(DRAIN_SEC / 0.5)):
                await asyncio.sleep(0.5)
                if last_done[0] >= base_sec + sent / 2 / SEND_RATE - 1.0:
                    break
        except Exception as error:   # noqa: BLE001
            print(f"    接続が切れました（{type(error).__name__}）。繋ぎ直します", flush=True)
        finally:
            listener.cancel()
    # 確定が取れたところまでを「送れた」とみなす。取れていない末尾は次の接続で送り直す
    #   （少し重なるが、重なりは落ちるより良い。重複は呼び出し側で落とす）
    confirmed = max(0, int((last_done[0] - base_sec) * SEND_RATE) * 2)
    return min(sent, confirmed) if confirmed else sent


async def stream(model: str, key: str, pcm: bytes, speed: float, speaker: str,
                 out_path: Path) -> list[dict]:
    """録音ぜんぶを流す。セッションの上限で切られたら繋ぎ直して続きから。"""
    rows: list[dict] = []
    latencies: list[float] = []
    total = len(pcm) / 2 / SEND_RATE
    position = 0
    sessions = 0

    while position < len(pcm):
        sessions += 1
        base_sec = position / 2 / SEND_RATE
        print(f"  接続 {sessions} 本目: {base_sec/60:.1f} 分から", flush=True)
        sent = await one_session(model, key, pcm[position:], base_sec, speed, speaker,
                                 rows, latencies)
        # 書けるたびに書く（落ちても途中までは残す。1 本目で全部失った 2026-09-13 の反省）
        out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                            encoding="utf-8")
        if sent <= 0:
            print("    1 バイトも送れませんでした。中断します", flush=True)
            break
        position += sent
        # 繋ぎ直しで少し重なる。同じ文が並んだら落とす
        deduped: list[dict] = []
        for row in rows:
            if deduped and row["text"] == deduped[-1]["text"] \
                    and abs(row["start_time"] - deduped[-1]["start_time"]) < 8:
                continue
            deduped.append(row)
        rows[:] = deduped
        print(f"    {position/2/SEND_RATE/60:.1f} 分ぶん送信ずみ / 確定 {len(rows)} 件", flush=True)

    valid = [value for value in latencies if 0 <= value < 30]
    if valid:
        print(f"  確定までの遅れ: 中央 {sorted(valid)[len(valid)//2]:.2f} 秒 / 最大 {max(valid):.2f} 秒")
    print(f"  音声 {total/60:.1f} 分 → 接続 {sessions} 本 / 確定 {len(rows)} 件")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.5-transcribe-live")
    parser.add_argument("--speaker", default="話者")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="実時間の何倍で流すか（1.0=実時間。会議中の挙動を見るなら 1.0）")
    parser.add_argument("--billing-project", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()

    ok, reason = billing_enabled(args.billing_project)
    print(f"課金の関門: {'✓' if ok else '✗'} {reason}")
    if not ok:
        return 1
    key = args.key_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()

    pcm = load_pcm16(args.audio, SEND_RATE)
    print(f"{args.audio.name}: {len(pcm)/2/SEND_RATE/60:.1f} 分を {args.speed:g} 倍速で流します")
    asyncio.run(stream(args.model, key, pcm, args.speed, args.speaker, args.out))
    print(f"書き出し: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
