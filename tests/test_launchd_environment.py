"""`.app`・launchd から起こされたときの環境の穴（2026-09-24・本番で発生）。

PATH: launchd の PATH は `/usr/bin:/bin:/usr/sbin:/sbin` だけ。ffmpeg・claude・ollama・gcloud が
  見えず、仕上げが止まり、最終パスが黙って別のエンジンへ落ちる。
画面収録: `.app` から起こすと許可先は Homebrew の Python.app。一覧から項目が消えると、
  「『Python』を許可して」と言われても**どこを足せばよいか分からない**まま会議が始まらなかった。
"""

from __future__ import annotations

import os

from src.app.user_path import ensure_user_path
from src.audio.sck_capture import permission_help, python_app_path


def test_launchd_path_gets_user_folders(tmp_path, monkeypatch):
    brew = tmp_path / "brew"
    brew.mkdir()
    monkeypatch.setattr("src.app.user_path.CANDIDATES", (str(brew), str(tmp_path / "missing")))
    env = {"PATH": "/usr/bin:/bin"}
    assert ensure_user_path(env) == [str(brew)]
    # 後ろに足す（その人の優先順を崩さない）・無いフォルダは足さない
    assert env["PATH"] == os.pathsep.join(["/usr/bin", "/bin", str(brew)])


def test_already_on_path_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr("src.app.user_path.CANDIDATES", (str(tmp_path),))
    env = {"PATH": f"{tmp_path}:/usr/bin"}
    assert ensure_user_path(env) == []
    assert env["PATH"] == f"{tmp_path}:/usr/bin"


def test_python_permission_help_shows_where_to_add_and_requests():
    asked = []
    text = permission_help("Python", request=lambda: asked.append(True),
                           app_path=lambda: "/x/Resources/Python.app")
    assert asked == [True]                     # 一覧に項目を載せ直すため、要求もする
    assert "/x/Resources/Python.app" in text
    assert "起動し直して" in text


def test_terminal_permission_help_has_no_python_path():
    text = permission_help("Terminal", request=lambda: None, app_path=lambda: "/x/Python.app")
    assert "Python.app" not in text and "『Terminal』" in text


def test_request_failure_still_gives_help():
    def boom():
        raise RuntimeError("no quartz")
    assert "『Python』" in permission_help("Python", request=boom, app_path=lambda: None)


def test_python_app_path_from_framework_prefix(tmp_path):
    app = tmp_path / "Resources" / "Python.app"
    app.mkdir(parents=True)
    assert python_app_path(str(tmp_path)) == str(app)
    assert python_app_path(str(tmp_path / "nope")) is None
