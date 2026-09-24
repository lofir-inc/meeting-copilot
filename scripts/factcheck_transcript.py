#!/usr/bin/env python3
"""記録済みの文字起こしにファクトチェック（検出＋ローカル判定）をかけて結果を見る。

本番に入れる前に、拾い方と判定の質を会議の外で確かめるための道具（Phase 7a・7b 段1）。

    python scripts/factcheck_transcript.py workspace/sessions/2026-09-11-2/transcripts.jsonl \\
        --window 60 --out workspace/eval/2026-09-11-gen/factcheck.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.factcheck import Claim, FactCheckConfig, FactChecker  # noqa: E402
from src.llm.ollama_client import LlmConfig, OllamaClient  # noqa: E402
from src.stt.whisper_client import TranscriptSegment  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def windows(segments: list[TranscriptSegment], width: float) -> list[list[TranscriptSegment]]:
    """会議秒で固定幅に区切る（空の窓は返さない）。"""
    groups: dict[int, list[TranscriptSegment]] = {}
    for segment in segments:
        groups.setdefault(int(segment.start_time // width), []).append(segment)
    return [groups[key] for key in sorted(groups)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("--config", type=Path, default=REPO / "config" / "settings.yaml")
    parser.add_argument("--window", type=float, default=60.0, help="何秒ぶんをまとめて見るか")
    parser.add_argument("--limit", type=int, help="先頭いくつの窓まで")
    parser.add_argument("--out", type=Path, help="結果の JSONL")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    values = config.get("meeting", {}).get("factcheck", {})
    cfg = FactCheckConfig(**{k: v for k, v in values.items() if k in FactCheckConfig.__dataclass_fields__})
    client = OllamaClient(LlmConfig(**config.get("llm", {})))
    if not client.health_check():
        raise SystemExit("Ollama に接続できません")
    checker = FactChecker(client, cfg)

    segments = [
        TranscriptSegment(**{k: row[k] for k in ("speaker", "text", "start_time", "end_time", "timestamp")})
        for row in (json.loads(line) for line in args.transcripts.read_text(encoding="utf-8").splitlines() if line.strip())
    ]
    found: list[Claim] = []
    started = time.perf_counter()
    for index, window in enumerate(windows(segments, args.window)):
        if args.limit is not None and index >= args.limit:
            break
        for claim in checker.detect(window):
            checker.verify(claim)
            found.append(claim)
            mark = {"一致": "✓", "要確認": "⚠", "不明": "?"}.get(claim.verdict, "?")
            print(f"  {mark} [{int(claim.at // 60)}:{int(claim.at % 60):02d}] {claim.kind:10s} {claim.speaker}: {claim.quote}")
            print(f"      {claim.verdict} — {claim.note}")
    elapsed = time.perf_counter() - started
    print(f"\n{len(found)} 件 / {elapsed:.0f} 秒（音声 {segments[-1].end_time / 60:.0f} 分ぶん）")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(claim.to_dict(), ensure_ascii=False) for claim in found) + "\n", encoding="utf-8")
        print(f"書き出しました: {args.out}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
