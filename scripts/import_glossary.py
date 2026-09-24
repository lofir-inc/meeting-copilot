#!/usr/bin/env python3
"""MacWhisper の置換辞書（Find & Replace）を config/glossary.yaml に取り込む。

MacWhisper 側で育てている語を正本にする。`~/Library/Preferences/com.goodsnooze.MacWhisper.plist`
の `globalReplaceList`（JSON 文字列の配列）を読み、`replacements:` として書き出す。

    python scripts/import_glossary.py                 # 差分を表示するだけ
    python scripts/import_glossary.py --write         # config/glossary.yaml を更新する

既存の glossary.yaml に手で足した語は消さない（MacWhisper 側と合わせて残す）。
"""

from __future__ import annotations

import argparse
import json
import plistlib
from pathlib import Path

import yaml

PLIST = Path.home() / "Library/Preferences/com.goodsnooze.MacWhisper.plist"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "config" / "glossary.yaml"


def read_macwhisper() -> dict[str, str]:
    """MacWhisper の置換辞書を {誤り: 正しい表記} で返す。"""
    if not PLIST.exists():
        raise SystemExit(f"MacWhisper の設定が見つかりません: {PLIST}")
    data = plistlib.loads(PLIST.read_bytes())
    table: dict[str, str] = {}
    for entry in data.get("globalReplaceList", []):
        try:
            item = json.loads(entry)
        except json.JSONDecodeError:
            continue
        original = str(item.get("original", "")).strip()
        replacement = str(item.get("replacement", "")).strip()
        if original and replacement and original != replacement:
            table[original] = replacement
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true", help="書き出す（指定しないと差分の表示だけ）")
    args = parser.parse_args()

    incoming = read_macwhisper()
    current: dict[str, str] = {}
    if args.out.exists():
        current = (yaml.safe_load(args.out.read_text(encoding="utf-8")) or {}).get("replacements") or {}

    added = {k: v for k, v in incoming.items() if k not in current}
    changed = {k: (current[k], v) for k, v in incoming.items() if k in current and current[k] != v}
    print(f"MacWhisper: {len(incoming)} 語 / いまの glossary.yaml: {len(current)} 語")
    print(f"  追加 {len(added)} 語 / 変更 {len(changed)} 語")
    for key, value in list(added.items())[:10]:
        print(f"    + {key} → {value}")
    for key, (before, after) in list(changed.items())[:10]:
        print(f"    ~ {key}: {before} → {after}")
    if not args.write:
        print("  （--write を付けると書き出します）")
        return 0

    merged = {**current, **incoming}
    body = yaml.safe_dump({"replacements": dict(sorted(merged.items()))}, allow_unicode=True, sort_keys=False, width=200)
    args.out.write_text(
        "# 文字起こしの結果を決まった表記へ置き換える辞書。\n"
        "# Whisper のヒント（initial_prompt）には使わない（2026-09-11: ヒントに入れた語は文中に紛れ込んだ）。\n"
        "# MacWhisper の Find & Replace から取り込む: python scripts/import_glossary.py --write\n"
        + body,
        encoding="utf-8",
    )
    print(f"  {len(merged)} 語を書き出しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
