"""録音を聞き直せる小ささに畳む（マイク別＋統合した 1 本）。

実測（2026-09-16・118 分の会議）: wav 1,299 MB → mp3 162 MB（8 分の 1・145 秒）。
"""

from __future__ import annotations

import subprocess

import pytest

from src.audio.archive import AUDIO_DIR, ArchiveConfig, compress, drop_wav, wav_bytes


def _recorded(tmp_path, *keys):
    for key in keys:
        (tmp_path / f"recording_{key}.wav").write_bytes(b"0" * 4096)
    return tmp_path


def _fake_ffmpeg(calls):
    def run(argv, **kwargs):
        calls.append(argv)
        out = argv[-1]
        from pathlib import Path

        Path(out).write_bytes(b"0" * 512)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return run


def test_マイク別と統合した1本を作る(tmp_path):
    _recorded(tmp_path, "self", "remote")
    calls = []

    made = compress(tmp_path, ArchiveConfig(), runner=_fake_ffmpeg(calls))

    assert set(made) == {"self", "remote", "meeting"}
    assert (tmp_path / AUDIO_DIR / "meeting.mp3").exists()
    merged = next(argv for argv in calls if argv[-1].endswith("meeting.mp3"))
    assert "amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11" in merged
    assert "-b:a" in merged and "64k" in merged


def test_片方しか無ければ統合しない(tmp_path):
    _recorded(tmp_path, "remote")

    made = compress(tmp_path, ArchiveConfig(), runner=_fake_ffmpeg([]))

    assert set(made) == {"remote"}


def test_もうあるものは作り直さない(tmp_path):
    _recorded(tmp_path, "self", "remote")
    calls = []
    compress(tmp_path, ArchiveConfig(), runner=_fake_ffmpeg(calls))
    first = len(calls)

    compress(tmp_path, ArchiveConfig(), runner=_fake_ffmpeg(calls))

    assert len(calls) == first          # 何度呼んでも増えない


def test_音量そろえを切れる(tmp_path):
    _recorded(tmp_path, "self", "remote")
    calls = []

    compress(tmp_path, ArchiveConfig(normalize=False), runner=_fake_ffmpeg(calls))

    merged = next(argv for argv in calls if argv[-1].endswith("meeting.mp3"))
    assert "loudnorm" not in " ".join(merged)


def test_ffmpegが失敗したら黙って通さない(tmp_path):
    _recorded(tmp_path, "remote")

    def broken(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Invalid data")

    with pytest.raises(RuntimeError):
        compress(tmp_path, ArchiveConfig(), runner=broken)


class TestDropWav:
    def test_mp3が無ければ捨てない(self, tmp_path):
        """「畳んだつもり」で消すと、録音からの作り直しができなくなる。"""
        _recorded(tmp_path, "self", "remote")

        with pytest.raises(RuntimeError):
            drop_wav(tmp_path, tmp_path / "trash")

        assert wav_bytes(tmp_path) > 0

    def test_畳んだあとはゴミ箱へ退避する(self, tmp_path):
        _recorded(tmp_path, "self", "remote")
        compress(tmp_path, ArchiveConfig(), runner=_fake_ffmpeg([]))

        moved = drop_wav(tmp_path, tmp_path / "trash")

        assert len(moved) == 2 and wav_bytes(tmp_path) == 0
        assert all(path.exists() for path in moved)
        assert "作り直すにはこの wav が要る" in (moved[0].parent / "WHY.md").read_text(encoding="utf-8")
