"""Silero VAD の最小 smoke テスト。"""

import numpy as np
import pytest

pytest.importorskip("silero_vad")

from src.audio.vad import SileroDetector


def test_silero_detects_sine_after_initial_silence():
    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate
    sine = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    waveform = np.concatenate((np.zeros(sample_rate), sine, np.zeros(sample_rate)))
    detector = SileroDetector(sample_rate, threshold=0.01)
    detections = [
        detector.is_voice(waveform[start : start + sample_rate // 10])
        for start in range(0, waveform.size, sample_rate // 10)
    ]
    assert not detections[0]
    assert any(detections[10:20])
