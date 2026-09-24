"""タスク管理 ブリッジの外部境界をモックで検査する。"""

import sys
from types import SimpleNamespace

from src.llm.meeting_state import Item
from src.task_hub import TaskHubBridge, TaskHubConfig


def _bridge(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "task_hub_context", SimpleNamespace(resolve_client=lambda name: {"client_id": "c1", "client_name": name, "task_db_id": "tasks", "minutes_db_id": "minutes"}))
    monkeypatch.setitem(sys.modules, "task_hub_minutes_lib", SimpleNamespace(register_tasks=lambda *args: calls.append(("register", args)) or [{"page_id": "p1", "url": "https://notion/p1"}]))
    monkeypatch.setitem(sys.modules, "task_hub_taskbrief", SimpleNamespace(set_task_status=lambda *args: calls.append(("status", args))))
    monkeypatch.setitem(sys.modules, "task_hub_notion", SimpleNamespace())
    return TaskHubBridge(TaskHubConfig(shared_dir="/does/not/exist")), calls


def test_registers_then_marks_in_progress(monkeypatch):
    bridge, calls = _bridge(monkeypatch)
    result = bridge.register_task(Item("T1", "資料を作る", due="10/1"), "brief", "2026-09-09", "自分")
    assert result == {"page_id": "p1", "url": "https://notion/p1"}
    assert [call[0] for call in calls] == ["register", "status"]
    task = calls[0][1][1][0]
    assert task["assignee"] == "自分"
    assert task["deadline"] == "2026-10-01"
    assert task["details"].endswith("（会議中に着手・realtime）")


def test_due_hints_cover_dates_and_words():
    due = TaskHubBridge.due_hint_of
    assert due("2026-10-01", "2026-09-09") == ("specific", "2026-10-01")
    assert due("1/1", "2026-09-09") == ("specific", "2027-01-01")
    assert due("今日", "2026-09-09") == ("specific", "2026-09-09")
    assert due("今週中", "2026-09-09")[0] == "this_week"
    assert due("再来週", "2026-09-09")[0] == "two_weeks"
    assert due("至急", "2026-09-09")[0] == "immediate"
    assert due("未定", "2026-09-09") == ("none", None)


def test_notify_posts_expected_body_and_can_be_disabled(monkeypatch):
    bridge, _ = _bridge(monkeypatch)
    sent = {}
    monkeypatch.setattr("src.task_hub.httpx.post", lambda url, **kwargs: sent.update(url=url, **kwargs) or SimpleNamespace(status_code=200))
    assert bridge.notify("開始") is True
    assert sent == {"url": bridge.cfg.chat_notify_url, "json": {"client_id": "c1", "text": "開始", "category": "minutes"}, "timeout": 10}
    disabled = TaskHubBridge(TaskHubConfig(slack=False))
    assert disabled.notify("送らない") is False
