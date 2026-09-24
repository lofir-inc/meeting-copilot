"""事前資料の固有名詞 — その会議だけの守り札と、要約・議事録への「正しい表記」。"""

from __future__ import annotations

import json

from src.text.glossary import apply_glossary
from src.text.terms import TERMS_FILE, extract_terms, load_terms, prompt_line, save_terms


def test_用語の節と会社名と英字の名前を先に拾う():
    text = ("# 打ち合わせ\n## 用語\n- Revuno（レブノ）\n- エムディクラウド株式会社\n"
            "本文: Revuno の有料プランと Findly 連携、Compa との比較。株式会社エムディクラウド")

    assert extract_terms(text) == ["Revuno", "エムディクラウド", "Findly", "Compa"]


def test_日本語にくっついた英字も取れる():
    """`\\b` だと「TORIAの」から TORIA が取れない（日本語の文字も単語の一部とみなされる）。"""
    assert "TORIA" in extract_terms("TORIAのOEM製品は産地で作る")


def test_番号と一般語は拾わない():
    terms = extract_terms("Q5 と Q6 の質問。サービスのデータとシステムのサービス。the plan")

    assert not {"Q5", "Q6", "サービス", "データ", "システム", "the"} & set(terms)


def test_1回しか出ないカタカナは拾わない():
    """カタカナは一般語が多い（実際の資料で ブランド・キーワード・ロジック が上位に来た）。"""
    assert extract_terms("インクジェットで刷る") == []
    assert extract_terms("インクジェットで刷る。インクジェットの色") == ["インクジェット"]


def test_まとめのLLMが挙げた語を先頭に置く():
    assert extract_terms("本文", extra=["取引先A"])[:1] == ["取引先A"]


def test_固有名詞を守り札にすると辞書が壊さない():
    """09-14 の会議: 社名「MDクラウド」が「クラウド→Claude」で「MDClaude」になった。"""
    pairs = [("クラウド", "Claude")]
    terms = extract_terms("## 用語\n- MDクラウド\n")

    assert apply_glossary("MDクラウドの方で", pairs) == "MDClaudeの方で"
    assert apply_glossary("MDクラウドの方で、クラウドで", pairs, terms) == "MDクラウドの方で、Claudeで"


def test_セッションに残して読み戻す(tmp_path):
    save_terms(tmp_path, ["Revuno", "TORIA"])

    assert load_terms(tmp_path) == ["Revuno", "TORIA"]
    assert json.loads((tmp_path / TERMS_FILE).read_text(encoding="utf-8")) == ["Revuno", "TORIA"]


def test_プロンプトの節は短く切る():
    line = prompt_line([f"語{i:03d}" for i in range(200)], max_chars=30)

    assert line.startswith("### この会議の固有名詞") and len(line.splitlines()[1]) <= 30
    assert prompt_line([]) == ""
