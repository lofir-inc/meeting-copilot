#!/usr/bin/env python3
"""文字起こしと正解を突き合わせて、置き換え辞書の候補を出す。

エンジンが変われば誤り方も変わる。`config/glossary.yaml` は MacWhisper / Whisper 特有の
誤りを集めたもので、**外（Gemini）のエンジンにはほとんど効かない**（2026-09-13 実測）。
エンジンごとに辞書を作り直すための道具（運用者 決定 2026-09-13）。

    python scripts/diff_glossary.py \\
        --transcripts workspace/eval/2026-09-11-gen/gemini-merged.jsonl \\
        --reference   workspace/eval/2026-09-11-gen/reference.jsonl

出すのは**候補**であって辞書そのものではない。必ず人が見てから `replacements` に入れること
（正解も自動の文字起こしなので、正解の側が間違っていることがある）。
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import unicodedata
from pathlib import Path

# 用語らしさ: 漢字・カタカナ・英数字を含むものだけ拾う（ひらがなだけの揺れは辞書にしない）
_TERM_LIKE = re.compile(r"[一-龥ァ-ヶー\w]")
_KANA_ONLY = re.compile(r"^[ぁ-んー、。\s]+$")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def pair_rows(ours: list[dict], reference: list[dict], window: float = 4.0,
              min_ratio: float = 0.5) -> list[tuple[str, str]]:
    """時刻が近く、文字がそこそこ似ている行どうしを対にする。"""
    pairs: list[tuple[str, str]] = []
    for row in ours:
        start = float(row["start_time"])
        best, score = None, 0.0
        for candidate in reference:
            if abs(float(candidate["start_time"]) - start) > window:
                continue
            ratio = difflib.SequenceMatcher(None, normalize(row["text"]),
                                            normalize(candidate["text"])).ratio()
            if ratio > score:
                best, score = candidate, ratio
        if best is not None and score >= min_ratio:
            pairs.append((normalize(row["text"]), normalize(best["text"])))
    return pairs


def collect(pairs: list[tuple[str, str]], max_len: int = 20) -> collections.Counter:
    """対になった行の中から「置き換わっている部分」だけを取り出して数える。"""
    counter: collections.Counter = collections.Counter()
    for ours, truth in pairs:
        matcher = difflib.SequenceMatcher(None, ours, truth)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            wrong, right = ours[i1:i2], truth[j1:j2]
            if not (1 <= len(wrong) <= max_len and 1 <= len(right) <= max_len):
                continue
            if wrong == right or not _TERM_LIKE.search(right):
                continue
            if _KANA_ONLY.match(wrong) and _KANA_ONLY.match(right):
                continue          # ひらがなだけの言い回しの揺れは辞書の仕事ではない
            counter[(wrong, right)] += 1
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=2, help="この回数以上出た誤りだけ出す")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--yaml", action="store_true", help="そのまま貼れる形で出す")
    args = parser.parse_args()

    pairs = pair_rows(load(args.transcripts), load(args.reference))
    counter = collect(pairs)
    found = [(wrong, right, count) for (wrong, right), count in counter.most_common()
             if count >= args.min_count][:args.top]

    if args.yaml:
        print("replacements:")
        for wrong, right, count in found:
            print(f"  {wrong}: {right}    # {count} 回")
        return 0

    print(f"対にできた行 {len(pairs)} / 候補 {len(found)} 件（{args.min_count} 回以上）\n")
    for wrong, right, count in found:
        print(f"  {count:3d}  {wrong!r} → {right!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
