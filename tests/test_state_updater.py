"""状態更新器のテスト。"""

from src.llm.meeting_state import MeetingState
from src.llm.state_updater import StateLoop, StateUpdater, StateUpdaterConfig, _validate_delta
from src.stt.whisper_client import TranscriptSegment


def response() -> dict:
    """有効な空差分を返す。"""
    return {"summary": "更新済み", "current_topic": "論点", "topic_changed": False, "new_decisions": [], "new_todos": [], "new_questions": [], "updates": [], "next_asks": []}


class FakeClient:
    """chat_json を記録するテスト用クライアント。"""

    def __init__(self, values: list[object]) -> None:
        self.values = values
        self.calls: list[str] = []
        self.kwargs: list[dict[str, object]] = []

    def chat_json(self, system: str, user: str, schema: dict, **kwargs: object) -> dict:
        """指定済みの結果を順番に返す。"""
        self.calls.append(user)
        self.kwargs.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value  # type: ignore[return-value]


def segment(start: float, text: str = "発話") -> TranscriptSegment:
    """テスト用発話を返す。"""
    return TranscriptSegment("田中", text, start, start + 1.0, "2026-01-01T00:00:00")


def test_update_success() -> None:
    """正常な JSON 差分を状態へ反映する。"""
    updater = StateUpdater(FakeClient([response()]), "system", "過去タスク", StateUpdaterConfig())
    state, stats = updater.update(MeetingState(), [segment(0)])
    assert state.summary == "更新済み"
    assert stats.parse_ok is True


def test_update_passes_think_to_chat_json() -> None:
    """状態更新は設定した think を JSON チャットへ渡す。"""
    client = FakeClient([response()])
    StateUpdater(client, "system", "", StateUpdaterConfig(think=False)).update(MeetingState(), [segment(0)])
    assert client.kwargs == [{"num_ctx": 8192, "num_predict": 700, "think": False}]


def test_empty_summary_keeps_previous_value() -> None:
    """空の要約は検証を通し、前回の要約を保持する。"""
    previous = MeetingState(summary="前の要約")
    empty_summary = response()
    empty_summary["summary"] = ""
    state, _ = StateUpdater(FakeClient([empty_summary]), "system", "", StateUpdaterConfig()).update(previous, [segment(0)])
    assert state.summary == "前の要約"


def test_invalid_json_retries_then_keeps_previous_state() -> None:
    """壊れた差分が続くと一度だけ再試行して前状態を返す。"""
    client = FakeClient([ValueError("bad json"), ValueError("bad json"), ValueError("bad json")])
    previous = MeetingState(summary="前の要約")
    state, stats = StateUpdater(client, "system", "", StateUpdaterConfig()).update(previous, [segment(0)])
    assert state.to_dict() == previous.to_dict()
    assert stats.parse_ok is False
    assert stats.retried is True
    assert len(client.calls) == 3


def test_three_attempts_succeed_after_two_failures() -> None:
    """三回目の応答が有効なら状態更新を成功として扱う。"""
    client = FakeClient([ValueError("bad json"), ValueError("bad json"), response()])
    state, stats = StateUpdater(client, "system", "", StateUpdaterConfig(max_attempts=3)).update(MeetingState(), [segment(0)])
    assert state.summary == "更新済み"
    assert stats.parse_ok is True
    assert stats.retried is True
    assert len(client.calls) == 3


def test_long_window_is_split() -> None:
    """会議秒の長いウィンドウを設定値で分割する。"""
    client = FakeClient([response(), response()])
    updater = StateUpdater(client, "system", "", StateUpdaterConfig(window_max_sec=120.0))
    _, stats = updater.update(MeetingState(), [segment(0), segment(121)])
    assert len(client.calls) == 2
    assert stats.parse_ok is True


def test_prompt_is_limited() -> None:
    """プロンプトは設定した最大文字数を超えない。"""
    updater = StateUpdater(FakeClient([]), "system", "過去" * 2000, StateUpdaterConfig(max_prompt_chars=200))
    assert len(updater.build_prompt(MeetingState(), [segment(0, "発話" * 1000)])) <= 200


def test_state_loop_flush_calls_hook() -> None:
    """flush は push 済み発話を処理してコールバックを呼ぶ。"""
    received: list[tuple[MeetingState, list[str]]] = []
    updater = StateUpdater(FakeClient([response()]), "system", "", StateUpdaterConfig())
    loop = StateLoop(updater, on_state=lambda state, stats, changes: received.append((state, changes)))
    loop.push(segment(0))
    state = loop.flush(1.0)
    assert state.summary == "更新済み"
    assert len(received) == 1


def test_state_loop_flush_finalizes_open_asks() -> None:
    """flush は会議終了の確定処理として、聞かれないまま残った問いを skipped にする。"""
    updater = StateUpdater(FakeClient([{**response(), "next_asks": ["予算の上限は？"]}]), "system", "", StateUpdaterConfig())
    loop = StateLoop(updater, on_state=lambda state, stats, changes: None)
    loop.push(segment(0))
    state = loop.flush(1.0)
    assert [(item.id, item.status) for item in state.asks] == [("A1", "skipped")]
    assert state.next_asks == []


def test_validate_delta_accepts_asked_update() -> None:
    """次に聞くことへ回答済みを記録する asked 更新を受け入れる。"""
    value = response()
    value["updates"] = [{"id": "A1", "status": "asked", "note": "回答"}]
    assert _validate_delta(value).updates == value["updates"]


def test_failed_part_keeps_earlier_parts_and_returns_leftover() -> None:
    """分割した後半で失敗しても、反映済みの前半は残し、後半の発話を持ち越し用に返す。"""
    client = FakeClient([response(), ValueError("bad"), ValueError("bad"), ValueError("bad")])
    updater = StateUpdater(client, "system", "", StateUpdaterConfig(window_max_sec=120.0))
    state, stats = updater.update(MeetingState(), [segment(0, "前半"), segment(121, "後半")])
    assert state.summary == "更新済み"
    assert stats.parse_ok is False
    assert [seg.text for seg in stats.leftover] == ["後半"]


def test_state_loop_carries_failed_segments_to_the_next_round() -> None:
    """2026-09-11 本番: Ollama が壊れていた 12 分の発話が要約から消えた。失敗した回の発話は次の回へ持ち越す。"""
    client = FakeClient([ValueError("bad")] * 3 + [response()])
    loop = StateLoop(StateUpdater(client, "system", "", StateUpdaterConfig()), on_state=lambda *args: None)
    loop.push(segment(0, "壊れていた間の発話"))
    loop._process(loop._take_buffer())
    loop.push(segment(30, "直ったあとの発話"))
    loop._process(loop._take_buffer())
    last_prompt = client.calls[-1]
    assert last_prompt.index("壊れていた間の発話") < last_prompt.index("直ったあとの発話")
    assert loop.state().summary == "更新済み"


def test_state_loop_gives_up_after_max_carry_rounds() -> None:
    """毎回壊れる窓を持ち越し続けると後ろが全部詰まるので、上限を超えたら捨てる。"""
    rounds = 3
    client = FakeClient([ValueError("bad")] * 3 * (rounds + 1))
    loop = StateLoop(StateUpdater(client, "system", "", StateUpdaterConfig(max_carry_rounds=rounds)), on_state=lambda *args: None)
    loop.push(segment(0))
    for _ in range(rounds + 1):
        loop._process(loop._take_buffer())
    assert loop._take_buffer() == []
