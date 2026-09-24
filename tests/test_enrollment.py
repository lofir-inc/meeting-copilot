"""話者登録時の声紋接近警告を検証する。"""

from types import SimpleNamespace

import numpy as np

from src.audio.devices import LevelReading
from src.audio.enrolled_diarizer import DiarizerConfig
from src.enrollment import EnrollmentConfig, _enroll_one


def _level() -> LevelReading:
    """録音成功扱いとなる入力レベルを返す。"""
    return LevelReading(peak_dbfs=-20.0, rms_dbfs=-30.0, seconds=5.0)


def _diarizer(similarity: float):
    """登録済み話者の近さを指定した偽 diarizer を返す。"""
    return SimpleNamespace(
        enrolled_names=["自分", "小島"],
        enroll=lambda name, audio, sample_rate: None,
        pairwise_similarity=lambda: {("自分", "小島"): similarity},
    )


class FakeFrameSource:
    """登録用に、ずっと話している音声を返す取得元。"""

    def read_seconds(self, seconds: float) -> np.ndarray:
        return np.full(int(seconds * 48000), 0.1, dtype=np.float32)


def test_enrollment_warns_when_voices_are_too_similar(monkeypatch, capsys):
    monkeypatch.setattr("src.enrollment.level_of", lambda *args: _level())
    config = EnrollmentConfig(diarizer=DiarizerConfig(enroll_warn_similarity=0.80))

    assert _enroll_one(_diarizer(0.83), "小島", FakeFrameSource(), config)

    assert "自分 と 小島 の声が似すぎています（0.83）" in capsys.readouterr().out


def test_enrollment_does_not_warn_when_voices_are_distinct(monkeypatch, capsys):
    monkeypatch.setattr("src.enrollment.level_of", lambda *args: _level())
    config = EnrollmentConfig(diarizer=DiarizerConfig(enroll_warn_similarity=0.80))

    assert _enroll_one(_diarizer(0.79), "小島", FakeFrameSource(), config)

    assert "声が似すぎています" not in capsys.readouterr().out


class ShortVoiceSource:
    """5 秒のうち最初の 0.4 秒だけ声が入った音声（2026-09-11 本番の 参加者B の登録）。"""

    def read_seconds(self, seconds: float) -> np.ndarray:
        audio = np.full(int(seconds * 48000), 1e-4, dtype=np.float32)
        audio[: int(0.4 * 48000)] = 0.1
        return audio


def test_enrollment_with_too_little_voice_asks_to_record_again(monkeypatch, capsys):
    monkeypatch.setattr("src.enrollment.level_of", lambda *args: _level())
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    enrolled: list[str] = []
    diarizer = SimpleNamespace(enrolled_names=[], enroll=lambda name, audio, sr: enrolled.append(name), pairwise_similarity=lambda: {})

    assert _enroll_one(diarizer, "参加者B", ShortVoiceSource(), EnrollmentConfig()) is False

    assert enrolled == []
    assert "声が入っていたのは 0.4 秒だけです" in capsys.readouterr().out


def test_voiced_seconds_counts_only_the_voice():
    from src.enrollment import voiced_seconds

    assert voiced_seconds(ShortVoiceSource().read_seconds(5.0), 48000) == 0.4
    assert voiced_seconds(FakeFrameSource().read_seconds(5.0), 48000) == 5.0
