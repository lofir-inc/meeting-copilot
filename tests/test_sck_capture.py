"""ScreenCaptureKit の実機に依存しない補助関数テスト。"""

from types import SimpleNamespace

from src.audio import sck_capture


def test_responsible_app_name_finds_first_app_parent(monkeypatch):
    results = iter([
        "101 /usr/local/bin/python\n",
        "1 /Applications/Agent.app/Contents/MacOS/Agent\n",
        "0 /sbin/launchd\n",   # 最も外側を採るため root まで遡る
    ])
    monkeypatch.setattr(sck_capture.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(results)))
    assert sck_capture.responsible_app_name() == "Agent"


def test_responsible_app_name_prefers_the_outermost_app(monkeypatch):
    """venv の python は Python.app の下にある。許可先は外側の Agent（bb 実測 2026-09-09）。"""
    results = iter([
        "300 /Library/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python\n",
        "200 /bin/zsh\n",
        "100 /Applications/Agent.app/Contents/Frameworks/Agent Helper.app/Contents/MacOS/Agent Helper\n",
        "0 /sbin/launchd\n",
    ])
    monkeypatch.setattr(sck_capture.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(results)))
    assert sck_capture.responsible_app_name() == "Agent"


def test_responsible_app_name_falls_back_to_executable_name(monkeypatch):
    results = iter(["1 /usr/local/bin/python\n", "0 /sbin/launchd\n"])
    monkeypatch.setattr(sck_capture.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=next(results)))
    assert sck_capture.responsible_app_name() == "python"


def test_has_screen_capture_permission_uses_quartz(monkeypatch):
    quartz = SimpleNamespace(CGPreflightScreenCaptureAccess=lambda: 1)
    monkeypatch.setattr(sck_capture, "Quartz", quartz)
    assert sck_capture.has_screen_capture_permission() is True
