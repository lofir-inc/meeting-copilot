"""`ffmpeg` が無いときに、**分かる形で止まる**こと。

なぜ要るか（2026-09-20・まっさらな Mac を想定した見直しで発見）: PATH から `ffmpeg` を外して
  仕上げを回したら、こう出た——

      FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
      （長いトレースバック）
      ✗ 議事録の材料がありません: …/minutes_input.md

  **本当の原因（brew install ffmpeg を忘れた）が、最後まで出てこない。**
  SETUP には書いてあるが、書いてあることと、抜けたときに分かることは別。

会議**中**は ffmpeg を使わない（録音は wav）。∴ 起動時は止めず、警告だけにする。
  会議の最中に落とさない、が優先。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio import ffmpeg


@pytest.fixture(autouse=True)
def _keep_path(monkeypatch):
    """available() は PATH を書き換えることがある。テストの外へ漏らさない。"""
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")


class TestRequire:
    def test_無ければ入れ方を出して止まる(self, monkeypatch):
        monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
        monkeypatch.setattr(ffmpeg, "SEARCH_DIRS", ())
        with pytest.raises(SystemExit) as stop:
            ffmpeg.require("会議のあとの仕上げ")
        message = str(stop.value)
        assert "会議のあとの仕上げ" in message      # どの作業で要るのか
        assert "brew install ffmpeg" in message     # どうすればいいのか
        assert "Traceback" not in message

    def test_あれば黙って通す(self, monkeypatch):
        monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: "/opt/homebrew/bin/ffmpeg")
        assert ffmpeg.require("会議のあとの仕上げ") is None


class TestWarn:
    def test_会議は止めない(self, monkeypatch):
        """ここで止めると、録れていた会議まで失う。"""
        monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
        monkeypatch.setattr(ffmpeg, "SEARCH_DIRS", ())
        said: list[str] = []
        assert ffmpeg.warn_if_missing(said.append) is False
        assert "brew install ffmpeg" in "".join(said)
        assert "会議は普通に回ります" in "".join(said)

    def test_あれば何も言わない(self, monkeypatch):
        monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: "/opt/homebrew/bin/ffmpeg")
        said: list[str] = []
        assert ffmpeg.warn_if_missing(said.append) is True
        assert said == []


class TestLaunchdPath:
    """2026-09-24 本番: 画面は launchd から PATH=/usr/bin:/bin:/usr/sbin:/sbin で起動され、
    /opt/homebrew/bin/ffmpeg があるのに「見つからない」で仕上げが止まった。"""

    def _fake_ffmpeg(self, folder: Path) -> None:
        tool = folder / "ffmpeg"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)

    def test_Homebrewの置き場所にあればPATHへ足して通す(self, tmp_path, monkeypatch):
        self._fake_ffmpeg(tmp_path)
        monkeypatch.setattr(ffmpeg, "SEARCH_DIRS", (str(tmp_path),))
        assert ffmpeg.available() is True
        assert ffmpeg.os.environ["PATH"].split(":")[0] == str(tmp_path)   # 後の ["ffmpeg", …] にも効く
        assert ffmpeg.require("会議のあとの仕上げ") is None

    def test_置き場所にも無ければ止まる(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ffmpeg, "SEARCH_DIRS", (str(tmp_path),))
        with pytest.raises(SystemExit):
            ffmpeg.require("会議のあとの仕上げ")
