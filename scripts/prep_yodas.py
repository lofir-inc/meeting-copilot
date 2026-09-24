#!/usr/bin/env python3
"""YODAS（日本語）から、比較用の音声と正解テキストを 1 組作る。

    ./scripts/prep_yodas.py --source workspace/bench/yodas --shard 00000000 --clips 120

なぜ YODAS か（2026-09-18）: Common Voice は**朗読**なので、3 つのエンジンが横並び（13.0〜13.3%）
  になり差が出なかった。YODAS は **YouTube の実際の会話**（対談・解説・雑談）で、雑音・言い淀み・
  重なりが入る。会議に近いのはこちら。
  `ja000` は**人が書いた字幕**（`ja100` は自動字幕なので正解には使えない）。
  ライセンスは **CC-BY-3.0**（表示すれば使える。外のエンジンへ出しても差し支えない）。

字幕は「話した通り」ではないことがある（要約・言い換え・書き言葉化）。だから**絶対値**は
  Common Voice より悪く出る。**エンジン同士の比較**に使う数字であって、正解率ではない。

準備:

    # 字幕（0.8MB）と音声（920MB）を 1 つぶんだけ落とす
    curl -L -o workspace/bench/yodas/00000000.txt  https://huggingface.co/datasets/espnet/yodas/resolve/main/data/ja000/text/00000000.txt
    curl -L -o workspace/bench/yodas/00000000.tar.gz https://huggingface.co/datasets/espnet/yodas/resolve/main/data/ja000/audio/00000000.tar.gz
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.prep_common_voice import GAP_SEC, build  # noqa: E402

OUT = REPO / "workspace" / "bench" / "yodas-ja"
MIN_SEC, MAX_SEC = 3.0, 15.0
DROP = re.compile(r"^[\s.…・]*$|^\[.*\]$")
"""空・記号だけ・[音楽] のような字幕は使わない。"""


def read_text(path: Path) -> dict[str, str]:
    """`<発話ID> <字幕>` の行を読む（Kaldi の形）。"""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, text = line.partition(" ")
        text = text.strip()
        if key and text and not DROP.match(text):
            found[key] = text
    return found


def seconds_of(utterance_id: str) -> float:
    """発話 ID の末尾（開始-終了。100 分の 1 秒）から長さを出す。"""
    parts = utterance_id.rsplit("-", 2)
    if len(parts) != 3 or not (parts[1].isdigit() and parts[2].isdigit()):
        return 0.0
    return (int(parts[2]) - int(parts[1])) / 100.0


def extract(archive: Path, wanted: set[str], into: Path) -> dict[str, Path]:
    """必要な音声だけを取り出す（920MB 全部は展開しない）。"""
    into.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key = Path(member.name).stem
            if key not in wanted or key in found:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            target = into / Path(member.name).name
            target.write_bytes(handle.read())
            found[key] = target
            if len(found) >= len(wanted):
                break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=REPO / "workspace" / "bench" / "yodas")
    parser.add_argument("--shard", default="00000000")
    parser.add_argument("--clips", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    texts = read_text(args.source / f"{args.shard}.txt")
    print(f"字幕: {len(texts)} 発話")

    # 動画を散らす（同じ動画ばかりだと、その話者・その収録環境に寄る）
    by_video: dict[str, list[str]] = {}
    for key in texts:
        by_video.setdefault(key.split("-")[0], []).append(key)
    videos = sorted(by_video)
    random.Random(args.seed).shuffle(videos)

    wanted: list[str] = []
    for video in videos:
        for key in sorted(by_video[video])[:3]:          # 1 本の動画から 3 発話まで
            if MIN_SEC <= seconds_of(key) <= MAX_SEC:
                wanted.append(key)
        if len(wanted) >= args.clips:
            break
    wanted = wanted[:args.clips]
    print(f"選んだ発話: {len(wanted)} 本 / 動画 {len({key.split('-')[0] for key in wanted})} 本")

    print("音声を取り出しています（必要なぶんだけ）…")
    files = extract(args.source / f"{args.shard}.tar.gz", set(wanted), args.out / "_clips")
    chosen = [{"path": str(files[key]), "sentence": texts[key], "seconds": round(seconds_of(key), 2),
               "speaker": key.split("-")[0]} for key in wanted if key in files]
    if not chosen:
        raise SystemExit("音声を取り出せませんでした（tar の中の名前が想定と違うかもしれません）")

    audio = build(chosen, args.out)
    reference = "".join(row["sentence"] for row in chosen)
    (args.out / "sample.txt").write_text(reference, encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in (args.out / "_clips").glob("*"):
        path.unlink(missing_ok=True)

    seconds = sum(row["seconds"] for row in chosen) + GAP_SEC * len(chosen)
    print(f"\n  音声 : {audio}（{seconds/60:.1f} 分・動画 {len({row['speaker'] for row in chosen})} 本）")
    print(f"  正解 : {args.out / 'sample.txt'}（{len(reference)} 文字）")
    print(f"\n  次: ./scripts/bench_asr.py --audio {audio} --reference {args.out / 'sample.txt'} "
          "--engines local,gemini,deepgram --allow-external")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
