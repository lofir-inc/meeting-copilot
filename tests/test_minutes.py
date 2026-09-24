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
