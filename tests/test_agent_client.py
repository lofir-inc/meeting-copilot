"""Agent CLI クライアントの引数と失敗を検査する。"""

from types import SimpleNamespace

import pytest

from src.agent_client import AgentClient, AgentError


def test_create_terminal_builds_safe_argv(monkeypatch):
    received = {}
    monkeypatch.setattr("src.agent_client.subprocess.run", lambda argv, **kwargs: received.update(argv=argv, **kwargs) or SimpleNamespace(returncode=0, stdout='{"id":"desk"}', stderr=""))
    result = AgentClient("agent-test", 12).create_terminal("/repo", "題", "cmd", focus=True)
    assert result == {"id": "desk"}
    assert received["argv"] == ["agent-test", "terminal", "create", "--worktree", "path:/repo", "--title", "題", "--command", "cmd", "--json", "--focus"]
    assert received["shell"] is False


def test_create_terminal_raises_on_nonzero(monkeypatch):
    monkeypatch.setattr("src.agent_client.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad"))
    with pytest.raises(AgentError, match="bad"):
        AgentClient().create_terminal("/repo", "題", "cmd")
