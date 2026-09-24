"""「外へ出すか」を画面で聞く経路のテスト（会議のあとの作り直しと、会議中の 30 秒刻みの両方）。

ここが緩むと、画面に出ないまま送る／返事を待ち続けて議事録ができない、のどちらかになる。
端末が無い環境でも回るよう、UI スレッドと bus は偽物を使う。
"""

import threading

import pytest

from src.stt.external_consent import ExternalSttConfig


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    def publish(self, name: str, payload: dict) -> None:
        self.published.append((name, payload))


class Asker:
    """`_ask_external_send` だけを取り出して動かすための最小の入れ物。"""

    from src.meeting_orchestrator import MeetingOrchestrator

    _ask_external = MeetingOrchestrator._ask_external
    _ask_external_send = MeetingOrchestrator._ask_external_send
    _ask_external_live = MeetingOrchestrator._ask_external_live
    answer_external_send = MeetingOrchestrator.answer_external_send

    def __init__(self, session_name: str, with_ui: bool) -> None:
        from pathlib import Path

        self.session_dir = Path(session_name)
        self.bus = FakeBus()
        self._ui_thread = object() if with_ui else None
        self._external_answer = None
        self._external_decided = threading.Event()


@pytest.fixture
def config():
    return ExternalSttConfig(enabled=True, billing_project="your-gcp-project", ask_timeout_sec=0.3)


def test_画面で送ると答えれば送る(config):
    asker = Asker("会議A", with_ui=True)
    threading.Timer(0.05, lambda: asker.answer_external_send("send")).start()

    assert asker._ask_external_send([], config) is True


def test_画面で手元と答えれば送らない(config):
    asker = Asker("会議A", with_ui=True)
    threading.Timer(0.05, lambda: asker.answer_external_send("keep")).start()

    assert asker._ask_external_send([], config) is False


def test_返事が無ければ送らない(config):
    """会議後の作り直しは無人で走ることがある。待ち続けると議事録ができない。"""
    asker = Asker("会議A", with_ui=True)

    assert asker._ask_external_send([], config) is False
    assert asker.bus.published[-1][1]["decision"] == "keep"


def test_聞いた内容が画面へ渡る(config):
    asker = Asker("会議A", with_ui=True)
    threading.Timer(0.05, lambda: asker.answer_external_send("keep")).start()
    asker._ask_external_send([], config)

    name, payload = asker.bus.published[0]
    assert name == "external_send"
    assert payload["phase"] == "ask"
    assert payload["session"] == "会議A"
    assert payload["project"] == "your-gcp-project"


def test_知らない答えは弾く(config):
    asker = Asker("会議A", with_ui=True)
    with pytest.raises(ValueError):
        asker.answer_external_send("たぶん")


# ---------------------------------------------------- 会議中（30 秒刻み）の確認

def test_会議中の送信も毎回聞く(config):
    """設定を on にしただけでは送らない。会議が始まる前に、その会議について聞く。"""
    asker = Asker("会議A", with_ui=True)
    threading.Timer(0.05, lambda: asker.answer_external_send("send")).start()

    assert asker._ask_external_live(config) is True
    payload = asker.bus.published[0][1]
    assert payload["live"] is True
    assert payload["files"] == ["会議中の音声（30 秒ごと）"]


def test_会議中も返事が無ければ送らない(config):
    asker = Asker("会議A", with_ui=True)

    assert asker._ask_external_live(config) is False
    assert asker.bus.published[-1][1]["decision"] == "keep"
