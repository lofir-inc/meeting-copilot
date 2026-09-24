"""ターミナルを出さずに会議を起こす（2026-09-18 運用者 指摘）。"""

from __future__ import annotations

import subprocess

from src.app import meeting_launch


class Fake:
    def __init__(self, code=None):
        self.code = code
        self.seen = None

    def popen(self, args, **kwargs):
        self.seen = (args, kwargs)
        return self

    def poll(self):
        return self.code


class TestStart:
    def test_切り離して起こす(self, tmp_path):
        """常駐アプリを終了しても会議は続く。"""
        fake = Fake()

        meeting_launch.start(tmp_path, tmp_path / "python", popen=fake.popen)
        args, kwargs = fake.seen

        assert args[1:] == ["-m", "src.main", "--mode", "meeting_loopback"]
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == str(tmp_path)

    def test_入力は渡さない(self, tmp_path):
        """端末が無いので、入力を待つ経路に落ちたらその場で終わってほしい。"""
        fake = Fake()

        meeting_launch.start(tmp_path, tmp_path / "python", popen=fake.popen)

        assert fake.seen[1]["stdin"] is subprocess.DEVNULL

    def test_出力をファイルに落とす(self, tmp_path):
        """黙って死なせない。切り離すと画面にも端末にも何も出ない。"""
        fake = Fake()

        meeting_launch.start(tmp_path, tmp_path / "python", popen=fake.popen)

        assert fake.seen[1]["stderr"] is subprocess.STDOUT
        assert meeting_launch.log_path(tmp_path).exists()

    def test_毎回まっさらにする(self, tmp_path):
        """前の回の失敗と混ざると読めない。"""
        meeting_launch.log_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        meeting_launch.log_path(tmp_path).write_text("前の回のログ\n", encoding="utf-8")
        fake = Fake()

        meeting_launch.start(tmp_path, tmp_path / "python", popen=fake.popen)

        assert "前の回" not in meeting_launch.log_path(tmp_path).read_text(encoding="utf-8")


class TestWhenItDoesNotStart:
    def test_落ちたら終了コードとログを見せる(self, tmp_path):
        meeting_launch.log_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        meeting_launch.log_path(tmp_path).write_text(
            "\n".join(f"行 {i}" for i in range(20)) + "\nModuleNotFoundError: sounddevice\n",
            encoding="utf-8")

        told = meeting_launch.why_not_started(Fake(code=1), tmp_path)

        assert "終了コード 1" in told
        assert "ModuleNotFoundError" in told
        assert "行 0" not in told               # 末尾だけ見せる

    def test_動いているのに画面が来なければそう言う(self, tmp_path):
        told = meeting_launch.why_not_started(Fake(code=None), tmp_path)

        assert "画面が" in told and "開きませんでした" in told

    def test_ログが無くても落ちない(self, tmp_path):
        assert "起動ログがありません" in meeting_launch.tail(tmp_path)
