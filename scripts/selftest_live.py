#!/usr/bin/env python3
"""会議**中**の経路（30 秒刻みで外へ投げる＝方式④）を、合成音声で通しで見る。

    ./scripts/selftest_live.py            # 送らずに刻みだけ見る（既定・ネットに出ない）
    ./scripts/selftest_live.py --send     # 実際に外へ投げる（会議ごとの確認と課金の確認を通す）

録音はどれも実在の相手の声なので試験には使えない。`make_test_session.py` の合成音声
（macOS の `say`）を使う。`--send` を付けたときだけ外へ出る。付けなければ 1 バイトも出ない。

見るのはこの 5 つ。精度ではなく**経路**:
  1. VAD の無音で窓が切れているか（語の途中で切っていないか）
  2. 1 分あたりのリクエスト数が上限（10/分）に収まるか
  3. `--send` で、会議ごとの確認と課金の確認を通ってから送っているか
  4. 返ってきた時刻が**会議の時計**に戻っているか（無音を落として繋いでいるため）
  5. 話者が手元の声紋で当たっているか（外の話者分けは使わない）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from replay_meeting import decode_audio  # noqa: E402

from src.audio.enrolled_diarizer import DiarizerConfig, EnrolledDiarizer  # noqa: E402
from src.audio.segment_labeler import label_segments  # noqa: E402
from src.audio.vad import ChunkBuilder, VadConfig  # noqa: E402
from src.stt.external_consent import ExternalSendRefused, ExternalSttConfig, approve, record_send  # noqa: E402
from src.stt.gemini_transcribe import TranscribeApi, TranscribeConfig, load_key  # noqa: E402
from src.stt.live_batch import (LiveBatchConfig, WindowAccumulator,  # noqa: E402
                                config_from as live_config_from, rows_in_meeting_time)

SESSION = "selftest-external"    # make_test_session.py が作るもの（使い回す）


def windows_of(audio: np.ndarray, sample_rate: int, vad: VadConfig, live: LiveBatchConfig,
               block_sec: float, key: str) -> list:
    """本番と同じ VAD → 窓の作り方に通す（壁時計の代わりに音声の進みを使う）。"""
    builder = ChunkBuilder(key, vad, sample_rate)
    accumulator = WindowAccumulator(key, sample_rate, live)
    windows, clock = [], 0.0
    block = int(block_sec * sample_rate)
    for index in range((audio.size + block - 1) // block):
        frame = audio[index * block:(index + 1) * block]
        if not frame.size:
            break
        clock += frame.size / sample_rate
        for chunk in builder.feed(frame):
            window = accumulator.feed(chunk, now=clock)
            if window is not None:
                windows.append(window)
        window = accumulator.tick(now=clock)
        if window is not None:
            windows.append(window)
    tail = accumulator.flush()
    if tail is not None:
        windows.append(tail)
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--send", action="store_true", help="実際に外へ投げる（既定は投げない）")
    parser.add_argument("--approved", action="store_true",
                        help="承認済み。人が別の画面で答えた直後にだけ付けるもので、手で付けるものではない"
                             "（課金の確認と送信の記録はそのまま通る）")
    parser.add_argument("--limit", type=int, default=2, help="送る窓の数（費用を抑えるため既定 2）")
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--config", type=Path, default=REPO / "config" / "settings.yaml")
    args = parser.parse_args()

    session = REPO / "workspace" / "sessions" / args.session
    remote_path = session / "recording_remote.wav"
    if not remote_path.exists():
        print(f"合成音声のセッションがありません。先に作ってください: "
              f"./scripts/make_test_session.py --name {args.session}")
        return 1

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    meeting = config.get("meeting", {})
    sample_rate = int(config.get("audio", {}).get("sample_rate", 48000))
    block_sec = float(config.get("audio", {}).get("chunk_duration_sec", 0.1))
    vad_values = {**config.get("vad", {}), **meeting.get("vad", {})}
    vad = VadConfig(**{k: v for k, v in vad_values.items() if k in VadConfig.__dataclass_fields__})
    external = ExternalSttConfig.from_mapping(meeting.get("external_stt"))
    # `auto` は数ではない。解決は live_batch.config_from に寄せてある（会議側と同じ経路）
    live = live_config_from(external.provider, external.live)

    audio = decode_audio(remote_path, sample_rate)
    windows = windows_of(audio, sample_rate, vad, live, block_sec, "remote")
    minutes = audio.size / sample_rate / 60
    print(f"\n相手側 {minutes:.1f} 分 → 窓 {len(windows)} 個"
          f"（{len(windows) / minutes:.1f} リクエスト/分・上限 10）")
    for index, window in enumerate(windows, 1):
        print(f"  {index}. 会議の {window.start_time:6.1f}〜{window.end_time:6.1f} 秒"
              f" → 送る音声 {window.speech_sec:4.1f} 秒（無音を落とした）")
    if not args.send:
        print("\n送っていません（--send で実際に投げます）。")
        return 0

    if not args.approved and not sys.stdin.isatty():
        print("✗ 端末から実行してください（画面が無いと承認を聞けないので、送らずに終わります）")
        return 1
    try:
        approval = approve(external, session, [remote_path],
                           ask=(lambda prompt: True) if args.approved else None)
    except ExternalSendRefused as refused:
        print(f"\n  送りません（{refused}）")
        return 0
    record_send(session, approval, model=external.model, note="selftest_live")

    api = TranscribeApi(load_key(external.key_file),
                        TranscribeConfig(model=external.model, timeout_sec=external.timeout_sec))
    diarizer_values = dict(meeting.get("diarizer", {}))
    diarizer = EnrolledDiarizer(DiarizerConfig(**{k: v for k, v in diarizer_values.items()
                                                  if k in DiarizerConfig.__dataclass_fields__}))
    for index, window in enumerate(windows[:args.limit], 1):
        started = time.time()
        result = api.transcribe_audio(window.audio, window.sample_rate, session / "external")
        rows = rows_in_meeting_time(window, result.get("utterances", []), result.get("words"))
        label_segments(rows, None, window.sample_rate, diarizer, slicer=window.slice)
        usage = result.get("usage", {})
        print(f"\n  窓 {index}（{window.start_time:.1f}〜{window.end_time:.1f} 秒）"
              f" 往復 {time.time() - started:.1f} 秒 / 入力 {usage.get('promptTokenCount', 0)} トークン")
        for row in rows:
            print(f"    [{row['start_time']:6.1f}] {row['speaker']}: {row['text']}")
        outside = [row for row in rows
                   if not (window.start_time - 0.5 <= row["start_time"] <= window.end_time + 0.5)]
        if outside:
            print(f"    ✗ 会議の時計から外れた行が {len(outside)} 件あります（時刻の戻しが壊れている）")
    print("\n見るところ: 時刻が窓の範囲に収まっているか／話者が 2 人に分かれているか／"
          "語の途中で切れていないか")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
