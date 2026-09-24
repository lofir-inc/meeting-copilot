#!/usr/bin/env python3
"""外部の文字起こしに、手元の声紋照合で話者を当て直す。

「文字起こしは外（Gemini）・話者の割り当ては手元」という組み合わせを測るための道具
（2026-09-13 運用者 方針）。音声は 1 バイトも外へ出さない — 読むのは手元の録音だけ。

    # 本番と同じ手順（登録状態から時刻順に逐次判定。不明話者の候補プールも通る）
    python scripts/relabel_speakers.py --mode online \
        --segments workspace/eval/2026-09-11-gen/gemini-remote.jsonl \
        --audio workspace/sessions/2026-09-11-2/recording_remote.wav \
        --speakers workspace/sessions/2026-09-11-2/speakers.json \
        --out workspace/eval/2026-09-11-gen/gemini-remote-ours.jsonl

    # 既にある声紋への最近傍照合だけ（上限の見積もり用）
    python scripts/relabel_speakers.py --mode centroid --names 参加者A,不明話者1 ...

出したものは `scripts/score_transcript.py` でそのまま採点できる。

区間が短いと embedding が不安定で判定できない（0.6 秒未満）。`--pad` で前後を伸ばすと
  判定できる区間が増える（Gemini の刻みは語単位まで細かいので効く。実測: 引き継ぎ 235→0 件）。
  それでも判定不能なものは**直前の話者を引き継ぐ**（無言で捨てない）。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.segment_labeler import label_segments  # noqa: E402
from src.audio.enrolled_diarizer import (  # noqa: E402
    DiarizerConfig,
    EnrolledDiarizer,
    MIN_DURATION_SEC,
    Speaker,
    peak_normalize,
    resample,
)

ENCODER_SAMPLE_RATE = 16000


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else -1.0


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def slice_chunk(audio: np.ndarray, rate: int, row: dict, pad: float) -> np.ndarray:
    start = max(0.0, float(row["start_time"]) - pad)
    end = min(len(audio) / rate, float(row["end_time"]) + pad)
    return audio[int(start * rate):int(end * rate)]


def run_centroid(rows, audio, rate, args) -> tuple[list[dict], collections.Counter]:
    """既にある声紋（speakers.json）への最近傍照合。"""
    source = EnrolledDiarizer.load(args.speakers, adapt=False)
    names = [name for name in args.names.split(",") if name]
    missing = [name for name in names if name not in source._speakers]
    if missing:
        raise SystemExit(f"speakers.json に居ない話者: {missing} / 居るのは {list(source._speakers)}")
    centroids = {
        name: (
            np.mean(source._speakers[name].enroll_embeddings, axis=0)
            if args.enroll_only
            else source._speakers[name].centroid(source.config.max_adapt_embeddings, source.config.enroll_weight)
        )
        for name in names
    }

    out: list[dict] = []
    counts: collections.Counter = collections.Counter()
    last = names[0]
    for row in rows:
        chunk = slice_chunk(audio, rate, row, args.pad)
        name = None
        if chunk.size / rate >= MIN_DURATION_SEC:
            wav = peak_normalize(resample(chunk, rate, ENCODER_SAMPLE_RATE))
            if wav is not None:
                embedding = EnrolledDiarizer.embed(wav, ENCODER_SAMPLE_RATE)
                if embedding is not None:
                    name = max(centroids, key=lambda key: cosine(embedding, centroids[key]))
        if name is None:
            name = last
            counts["引き継ぎ"] += 1
        else:
            last = name
        counts[name] += 1
        out.append({**row, "speaker": name})
    return out, counts


def run_online(rows, audio, rate, args) -> tuple[list[dict], collections.Counter]:
    """本番と同じ経路（登録状態から時刻順に identify）。中身は `src.audio.segment_labeler`。"""
    diarizer = EnrolledDiarizer(DiarizerConfig())
    if args.speakers and not args.no_enroll:
        source = EnrolledDiarizer.load(args.speakers, adapt=False)
        for name in args.names.split(","):
            if name in source._speakers:
                diarizer._speakers[name] = Speaker(
                    name=name,
                    enrolled=True,
                    enroll_embeddings=list(source._speakers[name].enroll_embeddings),
                )

    out = [dict(row) for row in rows]
    counts = label_segments(out, audio, rate, diarizer, pad_sec=args.pad)
    print(f"最終の登録簿: {diarizer.speaker_names}", file=sys.stderr)
    return out, collections.Counter(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["online", "centroid"], default="online")
    parser.add_argument("--segments", type=Path, required=True, help="話者を当て直したい jsonl")
    parser.add_argument("--audio", type=Path, required=True, help="その区間に対応する手元の録音")
    parser.add_argument("--speakers", type=Path, help="speakers.json")
    parser.add_argument("--names", default="参加者A,参加者B", help="使う話者（カンマ区切り）")
    parser.add_argument("--no-enroll", action="store_true", help="online: 登録なしで始める")
    parser.add_argument("--enroll-only", action="store_true", help="centroid: 会議中の適応を捨て登録時の声だけを使う")
    parser.add_argument("--pad", type=float, default=0.4, help="区間の前後に足す秒数")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "centroid" and not args.speakers:
        raise SystemExit("--mode centroid には --speakers が要る")

    rows = load_rows(args.segments)
    audio, rate = load_audio(args.audio)
    print(f"音声 {len(audio) / rate:.0f} 秒 / {rate} Hz、区間 {len(rows)} 件", file=sys.stderr)

    runner = run_online if args.mode == "online" else run_centroid
    out, counts = runner(rows, audio, rate, args)

    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out), encoding="utf-8")
    print(dict(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
