"""会議のあとの議事録作り — 使える手段を上から選び、駄目なら次へ落ちる。

誰の環境でも最後まで行けること（Claude のサブスクが無い・API キーが無い・オフライン）。
最後の段は「貼り付け用の指示書」で、何も入っていなくても議事録まで人の手 1 回で行ける。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.minutes import (
    PROMPT_FILE,
    Availability,
    MinutesConfig,
    MinutesMaker,
    resolve_order,
    split_transcript,
)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class Fake:
    def __init__(self, reply="# 議事録: 見積の確認\n\n## 決定事項\n- 見積は 9/20 まで", error=None):
        self.calls: list[tuple] = []
        self._reply = reply
        self._error = error

    def generate_text(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._error:
            raise self._error
        return self._reply


@pytest.fixture
def session(tmp_path):
    (tmp_path / "minutes_input.md").write_text(
        "# 議事録入力\n\n## 話者つき全文\n田中: 見積は 9/20 までに出します\n\n## 最終状態\n## 決定事項\n- 見積",
        encoding="utf-8")
    return tmp_path


def test_使える手段を上から選ぶ():
    order = resolve_order(MinutesConfig(), Availability(claude_cli=True, gemini=False, local=True))
    assert order == ["claude_cli", "local", "none"]


def test_何も無くても最後は指示書で終わる():
    """Claude のサブスクも API キーも Ollama も無い人でも、どこかで必ず止まる。"""
    assert resolve_order(MinutesConfig(), Availability()) == ["none"]


def test_手段を指定したらそれで駄目なら指示書():
    assert resolve_order(MinutesConfig(engine="local"), Availability(claude_cli=True)) == ["local", "none"]


def test_Claude_CLI_で作る(session):
    claude = Fake()
    engine, path = MinutesMaker(session, PROMPTS, claude=claude).make(["claude_cli", "none"])
    assert engine == "claude_cli"
    assert path.read_text(encoding="utf-8").startswith("# 議事録: 見積の確認")
    assert "見積は 9/20 までに出します" in claude.calls[0][0][0]     # 材料を渡している


def test_失敗したら次の手段へ落ちる(session):
    claude = Fake(error=RuntimeError("サブスクの枠を使い切った"))
    local = Fake(reply="# 議事録: 手元で作った")
    engine, path = MinutesMaker(session, PROMPTS, claude=claude, local=local).make(["claude_cli", "local", "none"])
    assert engine == "local"
    assert "手元で作った" in path.read_text(encoding="utf-8")


def test_全部駄目なら貼り付け用の指示書を置く(session):
    engine, path = MinutesMaker(session, PROMPTS, local=Fake(error=ValueError("空"))).make(["local", "none"])
    assert engine == "none" and path.name == PROMPT_FILE
    text = path.read_text(encoding="utf-8")
    assert "議事録を作るアシスタント" in text and "見積は 9/20 までに出します" in text   # 指示と材料が 1 つに


def test_手元の_LLM_は長いと分けてからまとめる(session):
    """手元のモデルは文脈が狭い。分けて要点を取り（map）、最後にまとめる（reduce）。"""
    long_body = "\n".join(f"田中: 発言 {index} " + "あ" * 50 for index in range(200))
    (session / "minutes_input.md").write_text(f"## 話者つき全文\n{long_body}\n\n## 最終状態\n- 決定",
                                               encoding="utf-8")
    local = Fake(reply="# 議事録: まとめ")
    MinutesMaker(session, PROMPTS, MinutesConfig(local_chunk_chars=3000), local=local).make(["local", "none"])
    assert len(local.calls) >= 3                            # 部分ごと＋最後のまとめ
    assert "部分ごとに要点" in local.calls[-1][0][1]         # 最後はまとめの呼び出し
    assert "## 最終状態" in local.calls[-1][0][1]            # 会議中の状態も最後に渡す


def test_全文は行の切れ目で分ける():
    chunks = split_transcript("\n".join(["あ" * 40] * 10), limit=100)
    assert all(len(chunk) <= 130 for chunk in chunks)
    assert "\n".join(chunks).count("あ" * 40) == 10          # 失わない


def test_材料が無ければ教える(tmp_path):
    with pytest.raises(FileNotFoundError):
        MinutesMaker(tmp_path, PROMPTS).make(["none"])


def test_前置きとコードブロックを外す(session):
    claude = Fake(reply="以下が議事録です。\n```markdown\n# 議事録: 本体\n```")
    _, path = MinutesMaker(session, PROMPTS, claude=claude).make(["claude_cli", "none"])
    assert path.read_text(encoding="utf-8").startswith("# 議事録: 本体")


@pytest.mark.parametrize("engine", ["auto", "gemini"])
@pytest.mark.parametrize("billing", ["paid", "unpaid", "error", "timeout", "project", "different", "env"])
def test_make_minutesも承認後にGoogleの課金とキーを確認する(session, tmp_path, monkeypatch, capsys, engine, billing):
    import subprocess
    import sys
    from types import SimpleNamespace
    from scripts import make_minutes as script

    key = tmp_path / "gemini.key"
    key.write_text("fake-key", encoding="utf-8")
    external = {"enabled": True, "provider": "deepgram", "billing_project": "fake-project",
                "key_file": str(key), "deepgram_key_file": "fake-deepgram.key"}
    gemini_values = {"key_file": str(key)}
    if billing == "project":
        external["billing_project"] = ""
    elif billing == "different":
        other = tmp_path / "other.key"
        other.write_text("fake-key", encoding="utf-8")
        gemini_values["key_file"] = str(other)
    elif billing == "env":
        gemini_values.clear()
        monkeypatch.setenv("GEMINI_API_KEY", "fake-env-key")
    monkeypatch.setattr(script, "load_config", lambda _: {
        "llm": {"gemini": gemini_values}, "meeting": {"external_stt": external}})
    monkeypatch.setattr(script.shutil, "which", lambda _: None)
    # Deepgram の口座が有効でも、Google の課金確認は別に必要。
    monkeypatch.setattr("src.stt.external_consent.paid_account", lambda _: (True, "deepgram paid"))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if billing == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 1 if billing == "error" else 0,
                                           stdout="True" if billing == "paid" else "False", stderr="offline")

    monkeypatch.setattr(subprocess, "run", fake_run)
    local = Fake(reply="# 議事録: 手元")
    local.health_check = lambda: True
    local.close = lambda: None
    monkeypatch.setattr(script, "OllamaClient", lambda _: local)
    gemini = Fake(reply="# 議事録: Gemini")
    gemini.usage = SimpleNamespace(calls=0)
    gemini.config = SimpleNamespace(model="fake-gemini")
    gemini.close = lambda: None
    made = []
    def make_gemini(config, *, verified_key):
        assert verified_key == "fake-key"
        made.append(config)
        return gemini

    monkeypatch.setattr(script, "GeminiClient", make_gemini)
    monkeypatch.setattr(script.known_people, "collect", lambda *a, **k: [])
    monkeypatch.setattr(script.clients, "load_cache", lambda *a: {})
    monkeypatch.setattr(script.clients, "people_of", lambda *a: [])
    import glossary_candidates
    monkeypatch.setattr(glossary_candidates, "refresh", lambda *a: (None, [], None))
    monkeypatch.setattr(sys, "argv", ["make_minutes.py", str(session), "--engine", engine, "--approved-external"])

    assert script.main() == 0
    assert bool(made) is (billing == "paid")
    assert bool(gemini.calls) is (billing == "paid")
    assert bool(local.calls) is (billing != "paid")
    assert (session / "external_sends.jsonl").exists() is (billing == "paid")
    if billing != "paid":
        assert "Gemini の課金を確かめられません" in capsys.readouterr().out
        assert "Gemini の課金を確かめられません" in (session / "interview.log").read_text(encoding="utf-8")
    if billing in {"project", "different", "env"}:
        assert calls == []


@pytest.mark.parametrize("case", ["paid", "unpaid", "refused", "env"])
def test_replay_stateも同じ関門を通り失敗したら手元へ(tmp_path, monkeypatch, case):
    import subprocess
    import sys
    from types import SimpleNamespace
    import yaml
    from scripts import replay_state as script

    monkeypatch.setattr(script, "_REPO", tmp_path)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "meeting_state_system.md").write_text("fake system", encoding="utf-8")
    transcripts = tmp_path / "transcripts.jsonl"
    transcripts.write_text("", encoding="utf-8")
    key = tmp_path / "gemini.key"
    key.write_text("fake-key", encoding="utf-8")
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump({
        "llm": {"gemini": {"key_file": str(key)} if case != "env" else {}},
        "meeting": {"external_stt": {"enabled": True, "provider": "deepgram", "key_file": str(key),
                                     "billing_project": "fake-project", "deepgram_key_file": "fake.key"}},
    }), encoding="utf-8")
    monkeypatch.setattr("src.stt.external_consent.paid_account", lambda _: (True, "deepgram paid"))
    monkeypatch.setattr(script, "ask_in_terminal", lambda *a: case != "refused")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-env-key")
    checked = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: checked.append(command) or
                        subprocess.CompletedProcess(command, 0, stdout="True" if case == "paid" else "False", stderr=""))
    made = []

    class FakeClient:
        def __init__(self, config, *, verified_key):
            assert verified_key == "fake-key"
            self.config = config
            made.append(config)

        def health_check(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(script, "GeminiClient", FakeClient)
    local = SimpleNamespace(health_check=lambda: True, close=lambda: None)
    monkeypatch.setattr(script, "OllamaClient", lambda config: local)
    used = []
    monkeypatch.setattr(script, "StateUpdater", lambda client, *a: used.append(client))
    monkeypatch.setattr(sys, "argv", ["replay_state.py", str(transcripts), "--session", "fake",
                                     "--config", str(config), "--llm", "gemini"])
    assert script.main() == 0
    assert bool(made) is (case == "paid")
    assert (used[0] is local) is (case != "paid")
    session = tmp_path / "workspace" / "sessions" / "fake"
    assert (session / "external_sends.jsonl").exists() is (case == "paid")
    if case != "paid":
        assert "Gemini は使いません" in (session / "interview.log").read_text(encoding="utf-8")
    if case in {"env", "refused"}:
        assert checked == []
