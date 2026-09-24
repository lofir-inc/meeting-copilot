"""用語辞書（認識後の置き換え）のテスト。"""

import pytest

from src.text.glossary import apply_glossary, load_glossary, load_protected


def _write(tmp_path, body: str):
    path = tmp_path / "glossary.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_file_is_empty(tmp_path):
    assert load_glossary(tmp_path / "none.yaml") == []


def test_longer_entries_are_replaced_first(tmp_path):
    """短い語を先に当てると長い語が壊れる。"""
    path = _write(tmp_path, "replacements:\n  エヌエイト: n8\n  エヌエイトエヌ: n8n\n")
    pairs = load_glossary(path)
    assert apply_glossary("エヌエイトエヌのワークフロー", pairs) == "n8nのワークフロー"


def test_real_world_entries(tmp_path):
    path = _write(tmp_path, "replacements:\n  クロード: Claude\n  お母さん: 自分さん\n")
    pairs = load_glossary(path)
    assert apply_glossary("あと自分のクロード", pairs) == "あと自分のClaude"
    assert apply_glossary("それをお母さんにパスして", pairs) == "それを自分さんにパスして"


def test_text_without_entries_is_untouched(tmp_path):
    pairs = load_glossary(_write(tmp_path, "replacements:\n  クロード: Claude\n"))
    assert apply_glossary("在庫の話に戻ります", pairs) == "在庫の話に戻ります"


def test_broken_file_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_glossary(_write(tmp_path, "replacements:\n  - クロード\n"))


def test_守り札の語の中では置き換えない(tmp_path):
    """「クラウド → Claude」は素で当てると「クラウドファンディング」を壊す。

    外（Gemini）の文字起こしは Claude を「クラウド」と書くので、この置き換えは要る。
    入れる前に守り札で潰した（2026-09-13）。
    """
    path = _write(tmp_path, "replacements:\n  クラウド: Claude\nprotect:\n  - クラウドファンディング\n")
    pairs, guards = load_glossary(path), load_protected(path)
    assert apply_glossary("クラウドで直す", pairs, guards) == "Claudeで直す"
    assert apply_glossary("クラウドファンディングの話", pairs, guards) == "クラウドファンディングの話"
    assert apply_glossary("クラウドファンディングをクラウドで作る", pairs, guards) == \
        "クラウドファンディングをClaudeで作る"


def test_守り札が無ければ従来どおり置き換える(tmp_path):
    path = _write(tmp_path, "replacements:\n  クラウド: Claude\n")
    assert load_protected(path) == []
    assert apply_glossary("クラウドで直す", load_glossary(path)) == "Claudeで直す"


def test_守り札がリストでなければ弾く(tmp_path):
    with pytest.raises(ValueError):
        load_protected(_write(tmp_path, "protect: クラウドファンディング\n"))


def test_長い語が先に置き換わる(tmp_path):
    """「クラウドコード」を「Claudeコード」にしない（短い語を先に当てると壊れる）。"""
    path = _write(tmp_path, "replacements:\n  クラウド: Claude\n  クラウドコード: Claude Code\n")
    assert apply_glossary("クラウドコードで直す", load_glossary(path)) == "Claude Codeで直す"


def test_長い語で整えたあと短い語がそこを触らない(tmp_path):
    """屋号「アトリエ・自社」を「アトリエ・自社」にしない。

    長い語（アトリエ自社 → アトリエ・自社）が先に当たったあと、短い語
    （自社 → 自社）が**その結果の中**を置き換えてしまう。守り札は
    「置き換えたあとの形」で書く必要がある（2026-09-13 の通しで見つけた形）。
    """
    path = _write(tmp_path, "replacements:\n"
                            "  アトリエ自社: アトリエ・自社\n"
                            "  自社: 自社\n"
                            "protect:\n"
                            "  - アトリエ・自社\n"
                            "  - アトリエ自社\n")
    pairs, guards = load_glossary(path), load_protected(path)
    assert apply_glossary("アトリエ自社の件", pairs, guards) == "アトリエ・自社の件"
    assert apply_glossary("自社の自分です", pairs, guards) == "自社の自分です"
    assert apply_glossary("アトリエ自社と自社", pairs, guards) == "アトリエ・自社と自社"
