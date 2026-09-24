#!/usr/bin/env python3
"""`gemini-3.5-transcribe-live` の口の形を実測で確かめる（合成音声だけを送る）。

    ./scripts/probe_live.py

推測で作らない。このリポジトリは既に「`generateContent` で動くと思ったら HTTP 200 で
本文が空」を踏んでいる（`gemini_transcribe.py` の冒頭）。Live 側も、繋がるか・設定の形・
返ってくるものの形を**先に見る**。

送るのは `workspace/sessions/selftest-external/recording_remote.wav`（macOS の `say` で
作った合成音声）。**クライアントの声は送らない。**
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.stt.external_consent import ExternalSttConfig, billing_enabled  # noqa: E402

HOST = "generativelanguage.googleapis.com"
PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
SEND_RATE = 16000          # Live へ送る PCM のサンプリングレート
FRAME_SEC = 0.2            # 1 回に送る長さ


def load_pcm16(path: Path, rate: int) -> bytes:
    """wav を rate Hz モノラル 16bit PCM にして返す（ffmpeg 任せ）。"""
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg に失敗: {result.stderr.decode()[:200]}")
    return result.stdout


def setup_variants(model: str) -> list[tuple[str, dict]]:
    """設定の形の候補。どれが通るか分からないので順に試す。"""
    full = f"models/{model}"
    return [
        ("① transcription_config つき", {"setup": {
            "model": full,
            "generation_config": {"transcription_config": {
                "language_codes": ["ja-JP"], "mode": {"type": "verbatim"}}},
        }}),
        ("② input_audio_transcription", {"setup": {
            "model": full,
            "input_audio_transcription": {},
        }}),
        ("③ model だけ", {"setup": {"model": full}}),
    ]


async def probe(model: str, key: str, audio: Path, seconds: float, realtime: bool) -> int:
    import websockets

    url = f"wss://{HOST}{PATH}?key={key}"
    pcm = load_pcm16(audio, SEND_RATE)[: int(seconds * SEND_RATE) * 2]
    frame = int(FRAME_SEC * SEND_RATE) * 2
    print(f"音源 {audio.name} の先頭 {seconds:.0f} 秒 / {len(pcm)/2/SEND_RATE:.1f} 秒ぶんの PCM\n")

    for label, setup in setup_variants(model):
        print(f"── {label} ──")
        try:
            async with websockets.connect(url, max_size=None, open_timeout=30) as socket:
                await socket.send(json.dumps(setup))
                try:
                    first = await asyncio.wait_for(socket.recv(), timeout=20)
                except asyncio.TimeoutError:
                    print("  setup の返事が来ません（20 秒）\n")
                    continue
                print(f"  setup の返事: {_short(first)}")
                if "error" in str(first).lower() and "setupComplete" not in str(first):
                    print()
                    continue

                started = time.perf_counter()
                received: list[str] = []

                async def listen() -> None:
                    try:
                        async for message in socket:
                            received.append(_short(message))
                            print(f"  [{time.perf_counter()-started:5.2f}s] {_short(message)}")
                    except Exception as error:   # noqa: BLE001
                        print(f"  受信終了: {type(error).__name__} {str(error)[:90]}")

                listener = asyncio.create_task(listen())
                for index in range(0, len(pcm), frame):
                    await socket.send(json.dumps({"realtime_input": {"audio": {
                        "data": base64.b64encode(pcm[index:index + frame]).decode(),
                        "mime_type": f"audio/pcm;rate={SEND_RATE}"}}}))
                    if realtime:
                        await asyncio.sleep(FRAME_SEC)
                await socket.send(json.dumps({"realtime_input": {"audio_stream_end": True}}))
                await asyncio.sleep(8)
                listener.cancel()
                print(f"  受け取ったメッセージ {len(received)} 件\n")
                if received:
                    return 0
        except Exception as error:   # noqa: BLE001
            print(f"  つながりません: {type(error).__name__} {str(error)[:160]}\n")
    return 1


def _short(message) -> str:
    text = message.decode() if isinstance(message, bytes) else str(message)
    return text if len(text) <= 320 else text[:320] + f"…（全 {len(text)} 文字）"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemini-3.5-transcribe-live")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--realtime", action="store_true", help="実時間で送る（既定は一気に送る）")
    parser.add_argument("--audio", type=Path,
                        default=REPO / "workspace/sessions/selftest-external/recording_remote.wav")
    args = parser.parse_args()

    import yaml

    settings = yaml.safe_load((REPO / "config" / "settings.yaml").read_text(encoding="utf-8")) or {}
    config = ExternalSttConfig.from_mapping(settings.get("meeting", {}).get("external_stt"))
    ok, reason = billing_enabled(config.billing_project)
    print(f"課金の関門: {'✓' if ok else '✗'} {reason}")
    if not ok:
        return 1
    key = Path(config.key_file).expanduser().read_text(encoding="utf-8").strip().splitlines()[0]

    if not args.audio.exists():
        raise SystemExit(f"音源がありません: {args.audio}（./scripts/make_test_session.py で作れます）")
    print(f"送るのは合成音声だけです: {args.audio}\n")
    return asyncio.run(probe(args.model, key, args.audio, args.seconds, args.realtime))


if __name__ == "__main__":
    raise SystemExit(main())
