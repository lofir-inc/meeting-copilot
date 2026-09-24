#!/usr/bin/env python3
"""文字起こしを「正解」JSONL と突き合わせて採点する。

正解は `scripts/macwhisper_reference.py` が作る（MacWhisper が同じ会議を別経路で録ったもの）。
正解も自動の文字起こしなので完璧ではない。数字は「基準との差」であって正解率ではない。
それでも設定を変えたときの **良し悪しの比較** には使える。

    python scripts/score_transcript.py \\
        --transcripts workspace/sessions/2026-09-11-2/transcripts.jsonl \\
        --reference workspace/eval/2026-09-11-gen/reference.jsonl

出すもの:
  文字の食い違い率 / 取りこぼし（正解にあってこちらに無い）/ 拾いすぎ（こちらにだけある）
  話者の一致率（文字が一致した行だけで数える。時刻の重なりだと同時発話に引っぱられる）

`--no-aizuchi` は相づちだけの行を**両側から**落としてから採点する。
  生の食い違い率はエンジンごとの「相づちをどれだけ 1 行として出すか」に強く引きずられる
  （2026-09-11 の会議: Gemini は 946 行中 428 行が相づちだが、文字数では 8.4% しかない）。
  設定やエンジンを比べるときは、こちらの数字で「中身の精度」を見る。
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import unicodedata
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text.aizuchi import is_aizuchi  # noqa: E402

_STRIP = re.compile(r"[\s、。,.!?！？…・「」『』（）()\-ー〜~]+")


@dataclass
class Row:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def normalized(self) -> str:
        return _STRIP.sub("", unicodedata.normalize("NFKC", self.text).lower())


def load(path: Path, shift: float = 0.0, drop_aizuchi: bool = False) -> list[Row]:
    """transcripts.jsonl / reference.jsonl を読む（shift 秒だけ時刻をずらす）。

    `drop_aizuchi` で相づちだけの行を落とす。落とすのは**両側**でないと意味がない
    （片側だけ落とすと、その分がまるごと食い違いに化ける）。
    """
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        text = str(data.get("text", ""))
        if drop_aizuchi and is_aizuchi(text):
            continue
        rows.append(Row(float(data["start_time"]) + shift, float(data["end_time"]) + shift,
                        str(data.get("speaker", "")), text))
    return sorted(rows, key=lambda row: row.start)


def edit_distance(hypothesis: str, reference: str) -> int:
    """文字単位の編集距離。"""
    previous = list(range(len(reference) + 1))
    for index, left in enumerate(hypothesis, 1):
        current = [index]
        for position, right in enumerate(reference, 1):
            current.append(min(previous[position] + 1, current[position - 1] + 1,
                               previous[position - 1] + (left != right)))
        previous = current
    return previous[-1]


def overlaps(row: Row, other: Row, pad: float = 0.7) -> bool:
    return min(row.end + pad, other.end) - max(row.start - pad, other.start) > 0


def best_match(row: Row, reference: list[Row], window: float = 4.0) -> tuple[Row | None, float]:
    """時刻が近い正解行のうち、文字がいちばん似ているものを返す。"""
    best: Row | None = None
    score = 0.0
    for candidate in reference:
        if abs(candidate.start - row.start) > window:
            continue
        ratio = difflib.SequenceMatcher(None, row.normalized, candidate.normalized).ratio()
        if ratio > score:
            best, score = candidate, ratio
    return best, score


def score(ours: list[Row], reference: list[Row], span: tuple[float, float], bucket_sec: float = 300.0) -> dict:
    """区間 span の中で採点して結果の辞書を返す。"""
    start, end = span
    ours = [row for row in ours if start <= row.start < end]
    reference = [row for row in reference if start <= row.start < end]
    buckets: list[dict] = []
    distance = reference_chars = hypothesis_chars = 0
    position = start
    while position < end:
        window = (position, min(position + bucket_sec, end))
        hypothesis = "".join(row.normalized for row in ours if window[0] <= row.start < window[1])
        truth = "".join(row.normalized for row in reference if window[0] <= row.start < window[1])
        if truth:
            delta = edit_distance(hypothesis, truth)
            distance += delta
            reference_chars += len(truth)
            hypothesis_chars += len(hypothesis)
            buckets.append({"start": window[0], "reference_chars": len(truth),
                            "our_chars": len(hypothesis), "cer": delta / len(truth)})
        position += bucket_sec

    missed = [row for row in reference if not any(overlaps(row, our) for our in ours)]
    extra = [row for row in ours if not any(overlaps(row, truth) for truth in reference)]

    pairs: collections.Counter = collections.Counter()
    for row in ours:
        if len(row.normalized) < 6:
            continue
        match, ratio = best_match(row, reference)
        if match is not None and ratio >= 0.6:
            pairs[(row.speaker, match.speaker)] += 1
    mapping: dict[str, str] = {}
    for (ours_name, reference_name), count in pairs.most_common():
        if ours_name not in mapping and reference_name not in mapping.values():
            mapping[ours_name] = reference_name
    matched = sum(pairs.values())
    correct = sum(count for (ours_name, reference_name), count in pairs.items() if mapping.get(ours_name) == reference_name)
    return {
        "cer": distance / reference_chars if reference_chars else 0.0,
        "reference_chars": reference_chars,
        "our_chars": hypothesis_chars,
        "buckets": buckets,
        "missed_lines": len(missed), "reference_lines": len(reference),
        "missed_chars": sum(len(row.normalized) for row in missed),
        "extra_lines": len(extra), "our_lines": len(ours),
        "extra_examples": [f"{row.start:.0f}s {row.speaker}「{row.text[:24]}」" for row in extra[:8]],
        "speaker_matched": matched, "speaker_correct": correct,
        "speaker_mapping": mapping,
        "speaker_pairs": {f"{a} → {b}": n for (a, b), n in pairs.most_common(12)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--shift", type=float, default=0.0,
                        help="こちらの時刻に足す秒数（2026-09-11 以前のデータは本編前に捨てた長さだけ手前にずれている）")
    parser.add_argument("--from-sec", type=float, default=0.0)
    parser.add_argument("--to-sec", type=float, default=float("inf"))
    parser.add_argument("--no-aizuchi", action="store_true",
                        help="相づちだけの行を両側から落としてから採点する（中身の精度を見る）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ours = load(args.transcripts, args.shift, args.no_aizuchi)
    reference = load(args.reference, 0.0, args.no_aizuchi)
    span = (args.from_sec, min(args.to_sec, max(row.end for row in reference)))
    result = score(ours, reference, span)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.no_aizuchi:
        print("相づちだけの行は両側から落として採点しています")
    print(f"区間 {span[0]:.0f}〜{span[1]:.0f} 秒 / 正解 {result['reference_lines']} 行・{result['reference_chars']} 字 "
          f"/ こちら {result['our_lines']} 行・{result['our_chars']} 字")
    print(f"  文字の食い違い率 : {result['cer'] * 100:.1f}%")
    print(f"  取りこぼし       : {result['missed_lines']} 行 / {result['missed_chars']} 字"
          f"（正解の {result['missed_chars'] / max(result['reference_chars'], 1) * 100:.1f}%）")
    print(f"  拾いすぎ         : {result['extra_lines']} 行（こちらの {result['extra_lines'] / max(result['our_lines'], 1) * 100:.1f}%）")
    for example in result["extra_examples"]:
        print(f"      {example}")
    if result["speaker_matched"]:
        print(f"  話者の一致       : {result['speaker_correct']}/{result['speaker_matched']} "
              f"({result['speaker_correct'] / result['speaker_matched'] * 100:.0f}%) 対応 {result['speaker_mapping']}")
        for pair, count in result["speaker_pairs"].items():
            print(f"      {pair}: {count}")
    print("  5 分ごと: " + " ".join(f"{bucket['cer'] * 100:.0f}%" for bucket in result["buckets"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
