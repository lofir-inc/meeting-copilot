#!/usr/bin/env python3
"""Common Voice（日本語）から、比較用の音声と正解テキストを 1 組作る。

    # 展開済みのフォルダを渡す（cv-corpus-…/ja/ の中に validated.tsv と clips/ がある）
    ./scripts/prep_common_voice.py --source ~/Downloads/cv-corpus-27.0-2026-xx-xx/ja --clips 120

    # できるもの
    workspace/bench/cv-ja/sample.wav   … つないだ音声（16kHz モノラル）
    workspace/bench/cv-ja/sample.txt   … 正解テキスト（読み上げた文を順につないだもの）
    workspace/bench/cv-ja/manifest.json… どのクリップを使ったか（話者・長さ・文）

そのまま `scripts/bench_asr.py` に渡せる:

    ./scripts/bench_asr.py --audio workspace/bench/cv-ja/sample.wav \\
        --reference workspace/bench/cv-ja/sample.txt --engines local,gemini --allow-external

Common Voice は **CC0**（パブリックドメイン）。外のエンジンへ出しても、クライアントの音声を
  出すことにはならない。だから `bench_asr.py` の関門（実会議は拒む）に引っかからない。

**朗読（Scripted Speech）である**ことを忘れない。実際の人・マイク・雑音・なまりは入るが、
  会議の難所（同時発話・言い淀み・相づち・遠いマイク）は測れない。第一次選抜まで。

**話者を散らす**（同じ人ばかりだと、その人の声に強いエンジンが有利になる）。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "workspace" / "bench" / "cv-ja"
MIN_SEC, MAX_SEC = 2.0, 12.0
GAP_SEC = 0.4
"""つなぎ目の無音。短すぎると語尾と語頭がくっつき、長すぎると VAD が別発話に割る。"""


def rows_of(source: Path) -> list[dict]:
    """`validated.tsv`（人が確かめたぶん）を読む。"""
    table = source / "validated.tsv"
    if not table.exists():
        raise SystemExit(f"validated.tsv がありません: {table}\n"
                         "  展開したフォルダの中の `ja` を指してください")
    with table.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def duration(path: Path) -> float:
    """音声の長さ（秒）。取れなければ 0（使わない）。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20)
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def pick(rows: list[dict], source: Path, wanted: int, seed: int) -> list[dict]:
    """話者を散らしてクリップを選ぶ（1 人 3 本まで）。"""
    random.Random(seed).shuffle(rows)
    per_speaker: dict[str, int] = {}
    chosen = []
    for row in rows:
        sentence = (row.get("sentence") or "").strip()
        name = (row.get("path") or "").strip()
        speaker = (row.get("client_id") or "")[:16]
        if not sentence or not name:
            continue
        if per_speaker.get(speaker, 0) >= 3:
            continue
        clip = source / "clips" / name
        if not clip.exists():
            continue
        seconds = duration(clip)
        if not (MIN_SEC <= seconds <= MAX_SEC):
            continue
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
        chosen.append({"path": str(clip), "sentence": sentence, "seconds": round(seconds, 2),
                       "speaker": speaker})
        if len(chosen) >= wanted:
            break
    if not chosen:
        raise SystemExit("使えるクリップが見つかりませんでした（clips/ の場所を確かめてください）")
    return chosen


def build(chosen: list[dict], out_dir: Path) -> Path:
    """つないで 1 本の wav にする（16kHz モノラル・間に無音）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / "_inputs.txt"
    silence = out_dir / "_gap.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", str(GAP_SEC), str(silence)], capture_output=True, check=False)
    pieces = []
    for index, row in enumerate(chosen):
        piece = out_dir / f"_piece-{index:04d}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", row["path"], "-ar", "16000", "-ac", "1", str(piece)],
                       capture_output=True, check=False)
        pieces += [piece, silence]
    listing.write_text("".join(f"file '{path}'\n" for path in pieces), encoding="utf-8")
    target = out_dir / "sample.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-ar", "16000", "-ac", "1", str(target)], capture_output=True, check=False)
    for path in set(pieces):
        path.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="展開した Common Voice の `ja` フォルダ")
    parser.add_argument("--clips", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260918, help="同じ種なら同じ組み合わせ（比較の再現用）")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = rows_of(args.source)
    print(f"validated.tsv: {len(rows)} 行。ここから {args.clips} 本を選びます（話者は 1 人 3 本まで）")
    chosen = pick(rows, args.source, args.clips, args.seed)
    audio = build(chosen, args.out)
    reference = "".join(row["sentence"] for row in chosen)
    (args.out / "sample.txt").write_text(reference, encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")

    seconds = sum(row["seconds"] for row in chosen) + GAP_SEC * len(chosen)
    print(f"\n  音声 : {audio}（{seconds/60:.1f} 分・{len({row['speaker'] for row in chosen})} 人）")
    print(f"  正解 : {args.out / 'sample.txt'}（{len(reference)} 文字）")
    print(f"  控え : {args.out / 'manifest.json'}")
    print(f"\n  次: ./scripts/bench_asr.py --audio {audio} "
          f"--reference {args.out / 'sample.txt'} --engines local,gemini --allow-external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
