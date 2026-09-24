"""ScreenCaptureKit の実機接続を診断するスクリプト。"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.audio.sck_capture import SckConfig, SckSystemAudio, has_screen_capture_permission, responsible_app_name

logger = logging.getLogger(__name__)


def write_tone(path: Path, sample_rate: int = 48000) -> None:
    """440 Hz、-12 dBFS、3 秒のモノラル WAV を生成する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    seconds = 3.0
    positions = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    samples = (10 ** (-12 / 20) * np.sin(2 * np.pi * 440 * positions)).astype(np.float32)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767).astype(np.int16).tobytes())


def analyse(samples: np.ndarray, sample_rate: int) -> tuple[float, float | None]:
    """ピーク dBFS と FFT 最大成分の周波数を返す。"""
    # -inf は厳密な JSON で読めない（aa 指摘 2026-09-09）。無音は -100.0 に丸め、dominant_hz は None にする
    if not len(samples):
        return -100.0, None
    peak = float(np.max(np.abs(samples)))
    peak_dbfs = -100.0 if peak == 0 else max(-100.0, 20 * np.log10(peak))
    if peak_dbfs <= -90.0:
        return peak_dbfs, None
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    if len(spectrum) <= 1:
        return peak_dbfs, None
    index = int(np.argmax(spectrum[1:]) + 1)
    return peak_dbfs, float(np.fft.rfftfreq(len(samples), 1 / sample_rate)[index])


def main() -> int:
    """SCK を指定秒数だけ実行し、診断結果を出力する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--tone", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "workspace" / "accept" / "sck_probe.wav")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--app", action="append", default=[], metavar="BUNDLE_ID")
    args = parser.parse_args()
    app_name = responsible_app_name()
    if not has_screen_capture_permission():
        print(f"システム設定 > プライバシーとセキュリティ > 画面収録 で『{app_name}』を許可してください。")
        return 2
    frames: list[np.ndarray] = []
    capture = SckSystemAudio(SckConfig(app_bundle_ids=list(args.app)), lambda frame: frames.append(frame.copy()))
    try:
        started_at = time.time()
        capture.start()
        if args.tone:
            tone_path = REPO / "workspace" / "accept" / "tone440.wav"
            write_tone(tone_path)
            subprocess.Popen(["afplay", str(tone_path)])
        time.sleep(args.seconds)
    except (PermissionError, RuntimeError) as error:
        print(f"SCK 初期化に失敗しました: {error}")
        print(f"システム設定 > プライバシーとセキュリティ > 画面収録 で『{app_name}』を許可してください。")
        return 2
    finally:
        capture.stop()
    tail = capture._reblocker.flush()
    if tail is not None:
        frames.append(tail)
    audio = np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(capture.cfg.sample_rate)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    stats = capture.stats()
    pts = stats.pop("buffer_pts")
    stats.pop("buffer_samples")
    intervals = np.diff(pts) * 1000 if len(pts) > 1 else np.empty(0)
    peak_dbfs, dominant_hz = analyse(audio, capture.cfg.sample_rate)
    result = {
        "permission": True,
        "responsible_app": app_name,
        "app_filter": args.app,
        "queue_mode": stats.pop("queue_mode"),
        "first_frame_latency_sec": None if capture.first_frame_at is None else capture.first_frame_at - started_at,
        "buffers": stats.pop("buffers"),
        "samples": int(len(audio)),
        "buffer_interval_ms": None if not len(intervals) else float(np.median(intervals)),
        "silent_gap_max_sec": stats.pop("gap_max_sec"),
        "gap_total_sec": stats.pop("gap_total_sec"),
        "peak_dbfs": peak_dbfs,
        "dominant_hz": dominant_hz,
        "pts_vs_samples_drift_sec": stats.pop("pts_vs_samples_drift_sec"),
        "format_flags": stats.pop("format_flags"),
    }
    print(f"ScreenCaptureKit probe: {result['buffers']} buffers, {result['samples']} samples")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
    if args.json:
        json_path = REPO / "workspace" / "accept" / "sck_probe.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
    return 0 if peak_dbfs > -60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
