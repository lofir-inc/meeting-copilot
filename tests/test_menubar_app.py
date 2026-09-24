"""メニューバー常駐アプリ（会議には触らない — 見るだけ）。"""

from __future__ import annotations

import plistlib
import stat
from datetime import datetime, timedelta, timezone

from src.app import bundle, login_item, running


class TestFindingAMeeting:
    def test_動いている会議を見つける(self):
        found = running.find_meeting(
            ports=[8765], is_open=lambda port: True,
            ask=lambda port: {"session_name": "2026-09-18_1400",
                              "started_at": "2026-09-18T14:00:00+09:00"})

        assert found is not None
        assert found.port == 8765 and found.session_name == "2026-09-18_1400"
        assert found.started_at is not None
        assert found.url == "http://127.0.0.1:8765/"

    def test_会議以外がポートを使っていても会議とみなさない(self):
        """2026-09-17 に別プロジェクトの http.server が 8765 に居座った。"""
        found = running.find_meeting(ports=[8765], is_open=lambda port: True,
                                     ask=lambda port: None)

        assert found is None

    def test_逃げた先のポートでも見つける(self):
        """8765 が塞がっていると隣へ逃げる（src/ui/server.py の PORT_TRIES）。"""
        seen = []

        def ask(port):
            seen.append(port)
            return {"session_name": "会議"} if port == 8768 else None

        found = running.find_meeting(ports=range(8765, 8775), is_open=lambda port: True, ask=ask)

        assert found is not None and found.port == 8768
        assert seen == [8765, 8766, 8767, 8768]     # 見つけたら止める

    def test_名前の無い応答は会議ではない(self):
        found = running.find_meeting(ports=[8765], is_open=lambda port: True,
                                     ask=lambda port: {"session_name": "  "})

        assert found is None

    def test_閉じているポートには聞きに行かない(self):
        """会議の前に毎回呼ぶので、開いていないポートで待たない。"""
        asked = []
        running.find_meeting(ports=[8765, 8766], is_open=lambda port: False,
                             ask=lambda port: asked.append(port))

        assert asked == []


class TestWhatTheMenuSays:
    def test_記録中は赤い印にする(self):
        assert running.title(running.Meeting(8765, "会議")) == "🔴"
        assert running.title(None) == "🎙"

    def test_経過時間を分で出す(self):
        jst = timezone(timedelta(hours=9))
        start = datetime(2026, 9, 18, 14, 0, tzinfo=jst)
        line = running.status_line(running.Meeting(8765, "さくら歯科", start),
                                   now=start + timedelta(minutes=10, seconds=5))

        assert "さくら歯科" in line and "10 分" in line

    def test_一時間を超えたら時間と分で出す(self):
        jst = timezone(timedelta(hours=9))
        start = datetime(2026, 9, 18, 14, 0, tzinfo=jst)
        line = running.status_line(running.Meeting(8765, "会議", start),
                                   now=start + timedelta(seconds=3725))

        assert "1 時間 2 分" in line

    def test_経過が取れなければ時間を出さない(self):
        line = running.status_line(running.Meeting(8765, "会議", None))

        assert line == "記録中: 会議" and "分" not in line

    def test_開始時刻が読めない形でも落ちない(self):
        """会議を止めないため、ここで例外を出さない。"""
        found = running.find_meeting(
            ports=[8765], is_open=lambda port: True,
            ask=lambda port: {"session_name": "会議", "started_at": "きのう"})

        assert found is not None and found.started_at is None
        assert running.status_line(found) == "記録中: 会議"


class TestLoginItem:
    def test_入れて外せる(self, tmp_path):
        loaded, unloaded = [], []
        app = tmp_path / "会議アシスタント.app"

        assert login_item.is_enabled(tmp_path) is False
        path = login_item.enable(app, tmp_path, load=loaded.append)

        assert login_item.is_enabled(tmp_path) is True
        assert loaded == [path]
        told = plistlib.loads(path.read_bytes())
        assert told["ProgramArguments"] == ["/usr/bin/open", "-a", str(app)]
        assert told["RunAtLoad"] is True and told["KeepAlive"] is False

        assert login_item.disable(tmp_path, unload=unloaded.append) is True
        assert unloaded == [path] and not path.exists()

    def test_入っていないものは外せない(self, tmp_path):
        assert login_item.disable(tmp_path, unload=lambda path: None) is False


class TestBundle:
    def test_開ける形になっている(self, tmp_path):
        python = tmp_path / "python"
        repo = tmp_path / "repo"
        app = bundle.build(python, repo, tmp_path)

        assert app.name == "会議アシスタント.app"
        told = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        assert told["CFBundleExecutable"] == "run"
        assert told["LSUIElement"] is True            # Dock に出さない

        binary = app / "Contents" / "MacOS" / "run"
        assert binary.stat().st_mode & stat.S_IXUSR
        text = binary.read_text(encoding="utf-8")
        assert str(python) in text and str(repo) in text

    def test_切り離した子として常駐を起こす(self, tmp_path):
        """2026-09-18 実測: Finder や open -a から起こしたプロセスがそのまま常駐すると、
        メニューバーに状態項目が置かれない（位置が x=0 のまま・isVisible は True を返す）。
        同じコードでもターミナルから直に起こすと置かれる。だから子にして切り離す。"""
        python = tmp_path / "venv" / "bin" / "python"
        repo = tmp_path / "repo"
        app = bundle.build(python, repo, tmp_path)
        text = (app / "Contents" / "MacOS" / "run").read_text(encoding="utf-8")

        assert text.startswith(f"#!{python}\n")            # シェバンで直に起こす
        assert "/bin/bash" not in text                     # シェルを挟まない
        assert "start_new_session=True" in text            # ここが要点
        assert "scripts/menubar_app.py" in text
        assert str(repo) in text

    def test_作り直せる(self, tmp_path):
        """置き場所や Python が変わったら作り直す。二重に作らない。"""
        first = bundle.build(tmp_path / "a", tmp_path / "repo", tmp_path)
        again = bundle.build(tmp_path / "b", tmp_path / "repo", tmp_path)

        assert first == again
        assert str(tmp_path / "b") in (again / "Contents" / "MacOS" / "run").read_text(encoding="utf-8")
        assert len(list(tmp_path.glob("*.app"))) == 1
