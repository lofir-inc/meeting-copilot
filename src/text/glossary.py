"""用語辞書 — 文字起こしの結果を、決まった表記へ置き換える。

Whisper へのヒント（initial_prompt）には入れない。2026-09-11 の本番で、ヒントに渡した語は
文中に紛れ込み（「GX」が 118 回）、無音区間ではヒントそのものを読み上げた。
ここは **認識したあとの文字を直す** 側で、誤認識を持ち込む余地がない。

辞書は `config/glossary.yaml`:

    replacements:
      エヌエイトエヌ: n8n
      アドバンストカスタムフィールズ: Advanced Custom Fields
    protect:
      - クラウドファンディング

`protect` は「この語の中にいるときは置き換えない」守り札（2026-09-13 に追加）。
外（Gemini）の文字起こしは Claude を **「クラウド」** と書く（Whisper は「クロード」）。
`クラウド: Claude` は要るが、素で置き換えると **「クラウドファンディング」→「Claudeファンディング」**
のような取り返しのつかない壊し方をする。守り札に載せた語の中では置き換えを見送る。
エンジンごとに誤り方が違うので辞書も分ける（`config/glossary-gemini.yaml`）。
実測（2026-09-11 の会議）: Whisper 用の 230 語は、Gemini の出力には **4 箇所しか当たらない**
（ローカル出力には 18 箇所）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_glossary(path: str | Path) -> list[tuple[str, str]]:
    """辞書を「長い語から先に置き換える」順に並べて返す。無ければ空。"""
    file_path = Path(path)
    if not file_path.exists():
        return []
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    table = data.get("replacements") or {}
    if not isinstance(table, dict):
        raise ValueError(f"{file_path}: replacements は「語: 置き換え後」の辞書で書いてください")
    pairs = [(str(key), str(value)) for key, value in table.items() if str(key)]
    # 短い語を先に置き換えると長い語が壊れる（「エヌエイト」→「エヌエイトエヌ」）
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    logger.info("用語辞書を読み込みました: %d 語 (%s)", len(pairs), file_path)
    return pairs


def load_protected(path: str | Path) -> list[str]:
    """守り札（この語の中では置き換えない）を返す。無ければ空。"""
    file_path = Path(path)
    if not file_path.exists():
        return []
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    guards = data.get("protect") or []
    if isinstance(guards, str) or not isinstance(guards, (list, tuple)):
        raise ValueError(f"{file_path}: protect は語のリストで書いてください")
    return [str(word) for word in guards if str(word)]


def apply_glossary(text: str, pairs: list[tuple[str, str]], protected: list[str] | tuple[str, ...] = ()) -> str:
    """辞書の語を置き換えた文字列を返す。

    `protected` に載った語の中にある一致は置き換えない
    （「クラウドファンディング」を「Claudeファンディング」にしないための守り札）。
    """
    for wrong, right in pairs:
        if wrong not in text:
            continue
        guards = [word for word in protected if wrong in word and word != wrong]
        text = _replace_outside(text, wrong, right, guards) if guards else text.replace(wrong, right)
    return text


def _replace_outside(text: str, wrong: str, right: str, guards: list[str]) -> str:
    """守り札の語に重なる一致だけを残して置き換える。"""
    spans: list[tuple[int, int]] = []
    for guard in guards:
        start = text.find(guard)
        while start != -1:
            spans.append((start, start + len(guard)))
            start = text.find(guard, start + 1)

    out: list[str] = []
    position = 0
    while True:
        found = text.find(wrong, position)
        if found == -1:
            out.append(text[position:])
            return "".join(out)
        end = found + len(wrong)
        inside = any(start <= found and end <= stop for start, stop in spans)
        out.append(text[position:found])
        out.append(wrong if inside else right)
        position = end
