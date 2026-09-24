"""Claude CLI のテキスト専用クライアントを検査する。"""

import json
from types import SimpleNamespace

import pytest

from src.llm.claude_cli_client import ClaudeCliClient
from src.llm.meeting_state import MeetingState


def _delta():
    return {"summary": "要約", "current_topic": "論点", "topic_changed": False, "new_decisions": [], "new_todos": [], "new_questions": [], "updates": [], "next_asks": []}


def test_available_is_false_without_claude(monkeypatch):
    monkeypatch.setattr("src.llm.claude_cli_client.shutil.which", lambda name: None)
    assert ClaudeCliClient().available() is False


def test_final_pass_validates_cli_json(monkeypatch):
    received = {}
    monkeypatch.setattr("src.llm.claude_cli_client.subprocess.run", lambda command, **kwargs: received.update(command=command, **kwargs) or SimpleNamespace(returncode=0, stdout=json.dumps({"result": json.dumps(_delta())}), stderr=""))
    result = ClaudeCliClient(model="test").final_pass(MeetingState(), "発話", "system")
    assert result.summary == "要約"
    assert received["command"] == ["claude", "-p", "--output-format", "json", "--model", "test"]


class TestExtractJson:
    """2026-09-12: 最終パスの応答がコードブロック付きで返り、json.loads が落ちた。"""

    def test_plain_json(self):
        from src.llm.claude_cli_client import extract_json

        assert extract_json('{"summary": "x"}') == {"summary": "x"}

    def test_fenced_json(self):
        from src.llm.claude_cli_client import extract_json

        assert extract_json('説明します。\n```json\n{"summary": "x"}\n```\n以上です。') == {"summary": "x"}

    def test_json_with_prose_around_it(self):
        from src.llm.claude_cli_client import extract_json

        assert extract_json('結果は次のとおりです。\n{"summary": "x"}\n') == {"summary": "x"}

    def test_no_json_raises(self):
        import pytest

        from src.llm.claude_cli_client import extract_json

        with pytest.raises(ValueError):
            extract_json("JSON はありません")


# ------------------------------------------------ 裏取り（Web 検索つき・オンデマンド）

def test_裏取りの応答を受け取る(monkeypatch):
    """出典は**実際に開いた URL** だけ。言葉（「公式サイト」等）は出典にしない。"""
    import subprocess

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["prompt"] = kwargs.get("input", "")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"result": json.dumps({
            "verdict": "要確認", "note": "2026-12-31 までの導入価格",
            "sources": ["https://ai.google.dev/gemini-api/docs/pricing", "公式サイト"]})}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ClaudeCliClient().verify("入力は 100 万トークン 0.75 ドル", "前後の発言")

    assert result == {"verdict": "要確認", "note": "2026-12-31 までの導入価格",
                      "sources": ["https://ai.google.dev/gemini-api/docs/pricing"]}
    assert "--allowedTools" in captured["command"]      # 検索して開けるようにして呼ぶ
    assert any("WebSearch" in part for part in captured["command"])
    assert "前後の発言" in captured["prompt"]


def test_知らない判定は弾く(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, stdout=json.dumps({"result": '{"verdict": "たぶん一致", "note": "", "sources": []}'}), stderr=""))
    with pytest.raises(ValueError, match="知らない判定"):
        ClaudeCliClient().verify("何かの発言")


def test_空の発言は投げない():
    with pytest.raises(ValueError):
        ClaudeCliClient().verify("   ")
