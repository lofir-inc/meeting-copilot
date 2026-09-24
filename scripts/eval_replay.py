#!/usr/bin/env python3
"""replay_result.json を再集計し、会議パイプラインの評価指標を返す。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


UNKNOWN_PREFIX = "不明話者"
UNKNOWN_QUESTION = f"{UNKNOWN_PREFIX}?"


def load_segments(path: Path) -> list[dict]:
    """replay_result.json から発話配列を読む。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"segments が配列ではありません: {path}")
    return segments


def _cap_hit_rate(segments: list[dict], max_chunk_sec: float) -> float:
    """VAD チャンクが上限で強制切断された割合。

    ターン分割後の JSON（`chunk_index` あり）では、ターンではなく **チャンク単位** で数える。
    ターンの長さで数えると、ターン分割の粗さ（None 穴の吸収など）で値が動き、
    VAD の挙動（＝Phase 1 の弱点 1）を測れない。旧 JSON（chunk_index 無し）は発話単位のまま。
    """
    if not segments:
        return 0.0
    if all("chunk_index" in segment for segment in segments):
        per_chunk: dict[int, float] = {}
        for segment in segments:
            index = int(segment["chunk_index"])
            per_chunk[index] = per_chunk.get(index, 0.0) + float(segment.get("duration") or 0.0)
        return sum(total >= max_chunk_sec - 0.1 for total in per_chunk.values()) / len(per_chunk)
    return sum(
        float(segment.get("duration") or 0.0) >= max_chunk_sec - 0.1 for segment in segments
    ) / len(segments)


def metrics(segments: list[dict], max_chunk_sec: float) -> dict:
    """計画で定義した replay 評価指標を返す。"""
    total_chars = sum(len(str(segment.get("text") or "")) for segment in segments)

    def chars_where(predicate) -> int:
        return sum(
            len(str(segment.get("text") or ""))
            for segment in segments
            if predicate(str(segment.get("speaker") or ""))
        )

    unknown_clusters: dict[str, dict[str, int]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or "")
        if speaker.startswith(UNKNOWN_PREFIX) and speaker != UNKNOWN_QUESTION:
            cluster = unknown_clusters.setdefault(speaker, {"utterances": 0, "chars": 0})
            cluster["utterances"] += 1
            cluster["chars"] += len(str(segment.get("text") or ""))

    unknown_chars = sum(cluster["chars"] for cluster in unknown_clusters.values())
    confident_similarities = [
        float(segment["similarity"])
        for segment in segments
        if segment.get("confident") and segment.get("similarity") is not None
    ]
    count = len(segments)
    return {
        "segments": count,
        "total_chars": total_chars,
        "empty_text_rate": (
            sum(not str(segment.get("text") or "") for segment in segments) / count if count else 0.0
        ),
        "named_char_rate": chars_where(lambda speaker: not speaker.startswith(UNKNOWN_PREFIX)) / total_chars if total_chars else 0.0,
        "unknown_n_char_rate": chars_where(lambda speaker: speaker.startswith(UNKNOWN_PREFIX) and speaker != UNKNOWN_QUESTION) / total_chars if total_chars else 0.0,
        "unknown_q_char_rate": chars_where(lambda speaker: speaker == UNKNOWN_QUESTION) / total_chars if total_chars else 0.0,
        "cap_hit_rate": _cap_hit_rate(segments, max_chunk_sec),
        "unknown_clusters": unknown_clusters,
        "largest_unknown_share": max((cluster["chars"] for cluster in unknown_clusters.values()), default=0) / unknown_chars if unknown_chars else 1.0,
        "unknown_clusters_ge": {
            5: sum(cluster["utterances"] >= 5 for cluster in unknown_clusters.values()),
            20: sum(cluster["utterances"] >= 20 for cluster in unknown_clusters.values()),
        },
        "confident_sim_median": median(confident_similarities) if confident_similarities else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="replay_result.json の評価指標を出す")
    parser.add_argument("replay_result", type=Path)
    parser.add_argument("--max-chunk-sec", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = metrics(load_segments(args.replay_result), args.max_chunk_sec)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
