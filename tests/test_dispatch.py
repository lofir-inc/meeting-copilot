"""会議中払い出しと議事録引き渡しを検査する。"""

import json
from pathlib import Path

from src.dispatch import DispatchConfig, DispatchResult, build_brief, dispatch, finish_command, write_minutes_handoff
from src.llm.meeting_state import Item, MeetingState
from src.stt.whisper_client import TranscriptSegment


class FakeBridge:
    """外部通信をせず各呼び出しを記録する タスク管理 の代役。"""

    def __init__(self, fail_register=False):
        self.fail_register = fail_register
        self.calls = []

    def register_task(self, todo, brief, meeting_date, owner):
        self.calls.append(("notion", owner, meeting_date))
        if self.fail_register:
            raise RuntimeError("notion down")
        return {"page_id": "page-1", "url": "https://notion/page-1"}

    def notify(self, text):
        self.calls.append(("slack", text))
        return True


class FakeAgent:
    """Agent を起動せず呼び出しだけを記録する代役。"""

    def __init__(self):
        self.calls = []

    def create_terminal(self, repo, title, command):
        self.calls.append((repo, title, command))
        return {"terminal": "one"}


def _state():
    todo = Item("T1", "提案書を作る", by="未定", due="今週中", created_at=100)
    return MeetingState(decisions=[Item("D1", "Aで進める"), Item("D2", "旧案", status="superseded", updated_at=99)], todos=[todo], questions=[Item("Q1", "予算は?", status="open")]), todo


def _segments():
    return [TranscriptSegment("自分", f"発話 {time}", time, time + 1, "") for time in (5, 10, 99, 130, 250)]


def _session():
    return {"client_name": "自社", "meeting_date": "2026-09-09", "session_name": "定例"}


def test_build_brief_has_seven_sections_and_only_context_segments():
    state, todo = _state()
    brief = build_brief(state, todo, _segments(), _session(), DispatchConfig(context_sec=10))
    for heading in ("# タスク", "## 担当・期日", "## 会議", "## 根拠の発話", "## 関連する決定", "## 未解決", "## Notion"):
        assert heading in brief
    assert "発話 99" in brief
    assert "発話 5" not in brief
    assert "発話 130" not in brief


def test_dispatch_keeps_four_routes_independent_and_deduplicates(tmp_path):
    state, todo = _state()
    bridge = FakeBridge(fail_register=True)
    agent = FakeAgent()
    cfg = DispatchConfig(agent_repo="/repo")
    result = dispatch(tmp_path, state, todo, _segments(), _session(), cfg, bridge, agent, "自分")
    assert result.brief_path.exists()
    assert "notion down" in result.errors["notion"]
    assert result.slack_ok is True
    assert agent.calls
    record = json.loads((tmp_path / "dispatch.jsonl").read_text(encoding="utf-8"))
    assert record["errors"]["notion"] == "notion down"
    again = dispatch(tmp_path, state, todo, _segments(), _session(), cfg, bridge, agent, "自分")
    assert again.already is True
    assert len(bridge.calls) == 2
    assert len(agent.calls) == 1


def test_handoff_applies_corrections_and_preserves_registered_task_name(tmp_path):
    state, todo = _state()
    segments = _segments()
    result = DispatchResult("T1", tmp_path / "dispatch" / "T1.md", notion={"page_id": "page-1", "url": "https://notion/page-1"})
    minutes_input, handoff = write_minutes_handoff(tmp_path, state, segments, _session(), [result], [{"start_time": 99.03, "new": "小島"}])
    content = minutes_input.read_text(encoding="utf-8")
    assert content.count(": 発話 ") == len(segments)
    assert "小島: 発話 99" in content
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    assert payload["registered_tasks"] == [{"page_id": "page-1", "url": "https://notion/page-1", "text": todo.text}]
    assert payload["transcript_path"] == str(minutes_input)


def test_finish_command_uses_handoff_date_and_repo(tmp_path):
    handoff = tmp_path / "minutes_handoff.json"
    handoff.write_text(json.dumps({"meeting_date": "2026-09-09"}), encoding="utf-8")
    argv = finish_command(handoff, DispatchConfig(minutes_repo="/minutes"))
    assert argv[:8] == ["agent", "terminal", "create", "--worktree", "path:/minutes", "--title", "議事録 2026-09-09", "--focus"]
    assert "/task_hub-minutes" in argv[-1]
    assert str(handoff) in argv[-1]
