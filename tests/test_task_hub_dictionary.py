"""タスク管理 のクライアント辞書とつなぐ（読む: キャッシュ／書く: 候補として）。"""

from __future__ import annotations

import json
from pathlib import Path

from src.text import task_hub_dictionary as md
from src.text.glossary_candidates import build_for_session


def pair(wrong, correct, status="確認済"):
    return {"wrong": wrong, "correct": correct, "status": status, "category": "VOCAB"}


class TestLivePairs:
    def test_会議中に当てるのは確認済の名前の対だけ(self):
        pairs = [
            pair("ITレビュー", "Revuno"),
            pair("ネクシス", "NEXIS"),
            pair("資料7の方", "資料請求フォーム"),      # 文脈で決まる直し。全行に当てると正しい発言を壊す
            pair("パープレキシティ", "Perplexity", status="候補"),
            pair("SARS", "SaaS"),                      # 略語は実在語と衝突する
            pair("ジシャ", "自社"),                    # 3 字以下は別の語に当たる
        ]

        assert md.live_pairs(pairs) == [("ITレビュー", "Revuno"), ("ネクシス", "NEXIS")]

    def test_他の誤りの一部になっている対は当てない(self):
        pairs = [pair("ネクシス", "NEXIS"), pair("ネクシスさん", "NEXISさん")]

        assert md.live_pairs(pairs) == [("ネクシスさん", "NEXISさん")]


def test_エンジンの辞書を優先して長い語から並べる():
    merged = md.merge_pairs([("クラウド", "Claude")], [("クラウド", "Cloud"), ("ITレビュー", "Revuno")])

    assert merged == [("ITレビュー", "Revuno"), ("クラウド", "Claude")]


def test_キャッシュを書いて読む(tmp_path):
    md.save_cache(tmp_path / "c.json", [pair("ネクシス", "NEXIS")], "自社")

    assert md.load_cache(tmp_path / "c.json")[0]["correct"] == "NEXIS"
    assert md.load_cache(tmp_path / "missing.json") == []


def _fake_tool(tmp_path: Path, script: str) -> Path:
    shared = tmp_path / "_shared"
    shared.mkdir()
    (shared / "dict_correct.py").write_text(script, encoding="utf-8")
    return shared


def test_タスク管理の道具で辞書を読む(tmp_path):
    shared = _fake_tool(tmp_path, (
        "def resolve_dictionary_db(client):\n    return 'db-' + client\n"
        "def load_pairs(db):\n    return [{'wrong': 'ネクシス', 'correct': 'NEXIS', 'status': '確認済'}]\n"))

    assert md.fetch(shared, "自社") == [{"wrong": "ネクシス", "correct": "NEXIS", "status": "確認済"}]


def test_選んだ対を候補として足す(tmp_path):
    shared = _fake_tool(tmp_path, (
        "import json, sys\n"
        "args = sys.argv\n"
        "print(json.dumps({'status': 'added', 'wrong': args[args.index('--add') + 1], 'client': args[args.index('--client') + 1]}))\n"))

    results = md.add_candidates(shared, "自社", [("レブノル", "Revuno")])

    assert results == [{"status": "added", "wrong": "レブノル", "client": "自社"}]


def test_タスク管理に既にある誤りは辞書の候補に出さない(tmp_path):
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text("replacements: {}\n", encoding="utf-8")
    rows = [{"speaker": "参加者C", "text": "レブノルとエムデイクラウド", "start_time": 0, "end_time": 1}]
    (tmp_path / "transcripts.jsonl").write_text(json.dumps(rows[0], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "glossary_llm.json").write_text(json.dumps([{"wrong": "レブノル", "right": "Revuno"}]), encoding="utf-8")

    _, candidates = build_for_session(tmp_path, glossary_path=glossary, self_name="自分",
                                      task_hub_pairs=[pair("レブノル", "Revuno", status="候補"),
                                                   pair("エムディクラウド", "エムディクラウド")])

    # タスク管理 が知っている正しい表記（エムディクラウド）は手がかりになる
    assert [(c.wrong, c.right) for c in candidates] == [("エムデイクラウド", "エムディクラウド")]
