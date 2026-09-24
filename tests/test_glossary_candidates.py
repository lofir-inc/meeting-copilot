"""置き換え辞書の候補 — 出すのは機械、入れるのは人。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.minutes import MinutesMaker
from src.text.glossary_candidates import (
    CANDIDATES_FILE,
    LLM_FILE,
    LLM_MARK,
    append_to_glossary,
    build_for_session,
    checked,
    from_llm,
    glossary_damage,
    glossary_for_session,
    near_misses,
    split_minutes,
)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def rows(*texts: str) -> list[dict]:
    return [{"speaker": "参加者C", "text": text, "start_time": float(i), "end_time": float(i + 1)}
            for i, text in enumerate(texts)]


class TestNearMisses:
    def test_知っている語に近いが違う表記を拾う(self):
        found = near_misses(rows("エムデイクラウドさんの件です"), {"エムディクラウド"})

        assert [(c.wrong, c.right) for c in found] == [("エムデイクラウド", "エムディクラウド")]

    def test_同じ表記とひらがなの揺れは拾わない(self):
        assert near_misses(rows("エムディクラウドの件"), {"エムディクラウド"}) == []

    def test_片方がもう片方を含むのは別の語(self):
        """実会議で Code→Codex・FTP→SFTP・mini→Gemini を出していた（全部誤り）。"""
        found = near_misses(rows("Claude Code で直す", "FTP で上げる", "Mac mini を買った"),
                            {"Codex", "SFTP", "Gemini"})

        assert found == []


class TestGlossaryDamage:
    def test_置き換えが別の語の一部を壊した跡を守り札の候補にする(self):
        """2026-09-14 の会議: 社名「MDクラウド」が「クラウド→Claude」で「MDClaude」になっていた。"""
        found = glossary_damage(rows("基本的にはMDClaudeの方で"), [("クラウド", "Claude")], [])

        assert [(c.kind, c.wrong) for c in found] == [("protect", "MDクラウド")]

    def test_辞書が正しい表記として知っている語は壊れていない(self):
        found = glossary_damage(rows("GitHubで繋ぐ"), [("ギット", "Git"), ("ギットハブ", "GitHub")], [])

        assert found == []

    def test_もう守り札にある語は出さない(self):
        assert glossary_damage(rows("MDClaude"), [("クラウド", "Claude")], ["MDクラウド"]) == []


class TestFromLlm:
    def test_全文に出てこない語は作り話として落とす(self):
        items = [{"wrong": "鳥居さん", "right": "取引先Aさん", "reason": "参加者名"},
                 {"wrong": "存在しない語", "right": "何か"}]

        found = from_llm(items, rows("鳥居さんに確認します", "鳥居さんから"))

        assert [(c.wrong, c.count) for c in found] == [("鳥居さん", 2)]

    def test_辞書に既にある語は出さない(self):
        assert from_llm([{"wrong": "鳥居さん", "right": "取引先Aさん"}], rows("鳥居さん"), existing={"鳥居さん"}) == []


def test_議事録の末尾の候補を本文から分ける():
    text = f"# 議事録: 見積\n\n## 概要\n- x\n\n{LLM_MARK}\n```json\n[{{\"wrong\": \"鳥居さん\", \"right\": \"取引先Aさん\"}}]\n```"

    body, items = split_minutes(text)

    assert LLM_MARK not in body and body.endswith("- x")
    assert items == [{"wrong": "鳥居さん", "right": "取引先Aさん"}]


def test_候補が壊れていても議事録は残る():
    body, items = split_minutes(f"# 議事録\n{LLM_MARK}\n[壊れた")

    assert body == "# 議事録" and items == []


def test_議事録と同じ呼び出しで候補を取っておく(tmp_path):
    (tmp_path / "minutes_input.md").write_text("## 全文\n参加者C: 鳥居さんに確認します\n", encoding="utf-8")

    class Claude:
        def generate_text(self, prompt, **kwargs):
            return f"# 議事録: 確認\n\n## 概要\n- 確認\n\n{LLM_MARK}\n[{{\"wrong\": \"鳥居さん\", \"right\": \"取引先Aさん\"}}]"

    engine, path = MinutesMaker(tmp_path, PROMPTS, claude=Claude()).make(["claude_cli", "none"])

    assert engine == "claude_cli" and LLM_MARK not in path.read_text(encoding="utf-8")
    assert json.loads((tmp_path / LLM_FILE).read_text(encoding="utf-8"))[0]["right"] == "取引先Aさん"


class TestAccept:
    def test_チェックしたものだけ読む(self, tmp_path):
        path = tmp_path / CANDIDATES_FILE
        path.write_text("- [x] 「鳥居さん」→「取引先Aさん」（2 回・議事録の LLM）\n"
                        "- [ ] 「鳥井さん」→「取引先Aさん」（1 回）\n"
                        "- [x] 守り札「MDクラウド」を足す（1 回・辞書が壊した跡）\n", encoding="utf-8")

        assert checked(path) == ([("鳥居さん", "取引先Aさん")], ["MDクラウド"])

    def test_コメントを残したまま辞書に足す(self, tmp_path):
        glossary = tmp_path / "glossary.yaml"
        glossary.write_text("# なぜこの辞書があるか\nreplacements:\n  クラウド: Claude   # 守り札が要る\n\n"
                            "protect:\n  - クラウドファンディング\n", encoding="utf-8")

        added, guards = append_to_glossary(glossary, [("鳥居さん", "取引先Aさん"), ("クラウド", "x")], "s1",
                                           ["MDクラウド"])

        text = glossary.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        assert added == [("鳥居さん", "取引先Aさん")] and guards == ["MDクラウド"]
        assert data["replacements"] == {"クラウド": "Claude", "鳥居さん": "取引先Aさん"}
        assert data["protect"] == ["クラウドファンディング", "MDクラウド"]
        assert "# 守り札が要る" in text and "s1" in text


def test_外で起こした会議は外のエンジンの辞書に足す(tmp_path):
    (tmp_path / "external_sends.jsonl").write_text(json.dumps({"note": "live_batch"}) + "\n", encoding="utf-8")
    meeting = {"glossary_path": "config/glossary.yaml", "external_stt": {"glossary_path": "config/glossary-gemini.yaml"}}

    assert glossary_for_session(tmp_path, meeting, Path("/repo")) == Path("/repo/config/glossary-gemini.yaml")
    (tmp_path / "external_sends.jsonl").unlink()
    assert glossary_for_session(tmp_path, meeting, Path("/repo")) == Path("/repo/config/glossary.yaml")


def test_会議1本ぶんの候補をファイルに書く(tmp_path):
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text("replacements:\n  クラウド: Claude\n", encoding="utf-8")
    lines = [{"speaker": "エムディクラウド 参加者C", "text": "MDClaudeの方で", "start_time": 0, "end_time": 1},
             {"speaker": "自分", "text": "エムデイクラウドさん", "start_time": 1, "end_time": 2}]
    (tmp_path / "transcripts.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in lines),
                                                 encoding="utf-8")

    path, candidates = build_for_session(tmp_path, glossary_path=glossary, self_name="自分")

    text = path.read_text(encoding="utf-8")
    assert "守り札「MDクラウド」" in text and "「エムデイクラウド」→「エムディクラウド」" in text


def test_同じ会議のぶんは見出しをまとめる(tmp_path):
    """1 語ずつ画面から入れると、1 件ごとに見出しが増えていた（2026-09-16 実機）。"""
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text("replacements:\n  クラウド: Claude\n", encoding="utf-8")

    append_to_glossary(glossary, [("ミナタ", "ミナト")], "2026-09-15_0957")
    append_to_glossary(glossary, [("お母さん", "自分さん")], "2026-09-15_0957")

    text = glossary.read_text(encoding="utf-8")
    assert text.count("会議のあとの候補から選んだもの") == 1
    assert yaml.safe_load(text)["replacements"] == {
        "クラウド": "Claude", "ミナタ": "ミナト", "お母さん": "自分さん"}
