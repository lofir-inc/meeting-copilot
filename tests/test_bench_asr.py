"""文字起こしエンジンの比較台（実会議の音声は外へ出さない）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import bench_asr


class TestGate:
    """関門: 実会議の音声を外のエンジンに渡さない。"""

    def test_セッションの音声は外へ出せない(self, tmp_path, monkeypatch):
        session_audio = bench_asr.REPO / "workspace" / "sessions" / "2026-09-17_0918" / "recording_remote.wav"

        with pytest.raises(SystemExit, match="実会議の音声"):
            bench_asr.refuse_real_meetings(session_audio, "gemini")

    def test_手元のエンジンなら止めない(self):
        session_audio = bench_asr.REPO / "workspace" / "sessions" / "x" / "recording_remote.wav"

        bench_asr.refuse_real_meetings(session_audio, "local")      # 例外が出なければよい

    def test_公開データは通す(self, tmp_path):
        bench_asr.refuse_real_meetings(tmp_path / "public.wav", "gemini")

    def test_許しを付けないと外のエンジンは走らない(self, tmp_path):
        audio = tmp_path / "public.wav"
        audio.write_bytes(b"RIFF")

        with pytest.raises(SystemExit, match="--allow-external"):
            bench_asr.bench(audio, "正解", ["gemini"], allow_external=False)


class TestScore:
    def test_同じ文なら食い違いゼロ(self):
        assert bench_asr.compare("こんにちは、今日はよろしく。", "こんにちは、今日はよろしく。")["error_rate"] == 0.0

    def test_違う分だけ数える(self):
        result = bench_asr.compare("耐水圧は2万ミリ", "耐水圧は20000mm")

        assert 0 < result["error_rate"] < 1.0

    def test_空なら全部外れ扱い(self, tmp_path, monkeypatch):
        audio = tmp_path / "public.wav"
        audio.write_bytes(b"RIFF")
        monkeypatch.setitem(bench_asr.ENGINES, "broken", lambda path: (_ for _ in ()).throw(RuntimeError("落ちた")))

        rows = bench_asr.bench(audio, "正解の文", ["broken"], allow_external=True)

        assert rows[0]["error_rate"] == 1.0 and "落ちた" in rows[0]["error"]


class TestCanonical:
    """数字と単位の書き方で順位が変わってしまうのを防ぐ（2026-09-18 に踏んだ）。"""

    def test_漢数字と算用数字をそろえる(self):
        assert bench_asr.canonical("三つの項目") == bench_asr.canonical("3つの項目")
        assert bench_asr.canonical("一点目") == bench_asr.canonical("1点目")

    def test_単位の書き方もそろえる(self):
        assert bench_asr.canonical("二万ミリメートル") == bench_asr.canonical("2万mm")

    def test_そろえると順位が変わる(self):
        """作り物の音声で Deepgram 1.2% / Whisper 7.1% と出たが、差は表記だけだった。"""
        reference = "決まった三つの項目について、二万ミリメートルという数値を"
        kanji = "決まった三つの項目について二万ミリメートルという数値を"
        arabic = "決まった3つの項目について2万mmという数値を"

        assert bench_asr.compare(kanji, reference)["error_rate"] == \
               bench_asr.compare(arabic, reference)["error_rate"]
