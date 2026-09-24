#!/usr/bin/env python3
"""文字起こしをローリング状態更新器へオフライン再生する。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.llm.meeting_state import MeetingState  # noqa: E402
from src.llm.gemini_client import GeminiClient, GeminiLlmConfig  # noqa: E402
from src.llm.ollama_client import LlmConfig, OllamaClient  # noqa: E402
from src.llm.state_updater import StateUpdater, StateUpdaterConfig  # noqa: E402
from src.stt.whisper_client import TranscriptSegment  # noqa: E402

logger = logging.getLogger(__name__)
PRICES = {
    # モデル: (入力 $/1M, 出力 $/1M)。2026-09-14 に https://ai.google.dev/gemini-api/docs/pricing で確認
    "gemini-3.8-flash": (0.75, 3.75),    # 2026-12-31 まで。2027-01-01 から 1.50 / 7.50
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "_unknown": (1.50, 9.00),            # 知らないモデルは高いほうで見積もる（驚かないように）
}

_REPO = Path(__file__).resolve().parent.parent


def load_segments(path: Path) -> list[TranscriptSegment]:
    """transcripts.jsonl を TranscriptSegment の時刻順リストで返す。"""
    segments: list[TranscriptSegment] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            segments.append(TranscriptSegment(**data))
    return sorted(segments, key=lambda value: value.start_time)


def windows(segments: list[TranscriptSegment], interval: float) -> list[list[TranscriptSegment]]:
    """会議秒で固定幅の非空ウィンドウへ発話を分ける。"""
    if not segments:
        return []
    result: list[list[TranscriptSegment]] = []
    start = 0.0
    pending = list(segments)
    while pending:
        end = start + interval
        current = [segment for segment in pending if start <= segment.start_time < end]
        pending = [segment for segment in pending if segment.start_time >= end]
        if current:
            result.append(current)
        start = end
    return result


def load_config(path: Path | None) -> StateUpdaterConfig:
    """設定 YAML の meeting.state を StateUpdaterConfig として読む。"""
    if path is None:
        return StateUpdaterConfig()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    values = data.get("meeting", {}).get("state", data.get("state", {}))
    allowed = set(StateUpdaterConfig.__dataclass_fields__)
    return StateUpdaterConfig(**{key: value for key, value in values.items() if key in allowed})


def load_prior_tasks(prep_dir: Path | None) -> str:
    """事前資料ディレクトリの Markdown をファイル名順に結合する。"""
    if prep_dir is None or not prep_dir.exists():
        return ""
    return "\n\n".join(
        content
        for markdown_path in sorted(prep_dir.glob("*.md"))
        if (content := markdown_path.read_text(encoding="utf-8").strip())
    )


def main() -> int:
    """コマンドラインから状態再生を実行する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prep-dir", type=Path)
    parser.add_argument("--llm-model", default=None,
                        help="エンジンのモデルを上書きする（例 gemini-3.8-flash）。"
                             "モデルを比べるときに使う。省略時は settings のまま")
    parser.add_argument("--llm", choices=["ollama", "gemini"], default="ollama",
                        help="要約を回すエンジン。gemini は settings の llm.gemini を読む"
                             "（送るのはテキストだけ・音声は送らない）")
    parser.add_argument(
        "--dump-prompts", action="store_true",
        help="窓ごとのプロンプト本文を <session>/prompts/NNN.txt に残す（不明話者? の扱いを目で見る用）",
    )
    args = parser.parse_args()
    if not args.transcripts.exists():
        print(f"transcripts.jsonl が見つかりません: {args.transcripts}", file=sys.stderr)
        return 2

    system_path = _REPO / "prompts" / "meeting_state_system.md"
    cfg = load_config(args.config)
    cfg.interval_sec = args.interval
    if args.llm == "gemini":
        values = {}
        if args.config and args.config.exists():
            with args.config.open(encoding="utf-8") as handle:
                values = ((yaml.safe_load(handle) or {}).get("llm", {}) or {}).get("gemini", {}) or {}
        known = set(GeminiLlmConfig.__dataclass_fields__)
        gemini_cfg = GeminiLlmConfig(**{k: v for k, v in values.items() if k in known})
        if args.llm_model:
            gemini_cfg.model = args.llm_model
        client = GeminiClient(gemini_cfg)
        if not client.health_check():
            print("Gemini に繋がりません（キーかモデル名を確認）", file=sys.stderr)
            client.close()
            return 2
        print(f"要約エンジン: Gemini（{client.config.model}）送るのはテキストだけ")
    else:
        local_cfg = LlmConfig()
        if args.llm_model:
            local_cfg.model = args.llm_model
        client = OllamaClient(local_cfg)
        if not client.health_check():
            print("Ollama に接続できません: http://localhost:11434", file=sys.stderr)
            client.close()
            return 2
    session_dir = _REPO / "workspace" / "sessions" / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    updater = StateUpdater(client, system_path.read_text(encoding="utf-8"), load_prior_tasks(args.prep_dir), cfg)
    state = MeetingState()
    try:
        # buffering=1: 1 行ごとに flush（途中で殺されても行の途中で切れた残骸を残さない）
        with (session_dir / "state_trace.jsonl").open("w", encoding="utf-8", buffering=1) as trace:
            for index, window in enumerate(windows(load_segments(args.transcripts), args.interval)):
                if args.limit_windows is not None and index >= args.limit_windows:
                    break
                if args.dump_prompts:
                    prompt_dir = session_dir / "prompts"
                    prompt_dir.mkdir(exist_ok=True)
                    (prompt_dir / f"{index:03d}.txt").write_text(
                        updater.build_prompt(state, window), encoding="utf-8"
                    )
                state, stats = updater.update(state, window)
                record = {
                    "window_index": index,
                    "t_start": window[0].start_time,
                    "t_end": window[-1].end_time,
                    "prompt_chars": stats.prompt_chars,
                    "latency_sec": stats.latency_sec,
                    "parse_ok": stats.parse_ok,
                    "retried": stats.retried,
                    "changes": stats.changes,
                    "state": state.to_dict(),
                }
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info("window=%d parse_ok=%s changes=%d", index, stats.parse_ok, len(stats.changes))
        state.finalize(state.updated_at)
        (session_dir / "state.json").write_text(state.to_json() + "\n", encoding="utf-8")
        (session_dir / "interview_summary.md").write_text(state.to_markdown("会議要約"), encoding="utf-8")
    finally:
        usage = getattr(client, "usage", None)
        if usage is not None and usage.calls:
            import statistics

            median = statistics.median(usage.latencies) if usage.latencies else 0.0
            # 単価はモデルごとに違う（2026-09-14 に公式の価格表で確認）。
            #   同じ式で別のモデルを見積もると、費用の比較が**そのまま間違う**。
            price_in, price_out = PRICES.get(client.config.model, PRICES["_unknown"])
            cost = usage.prompt_tokens / 1e6 * price_in + usage.output_tokens / 1e6 * price_out
            print(f"\n呼び出し {usage.calls} 回（失敗 {usage.failures}）"
                  f" / 入力 {usage.prompt_tokens:,} ・出力 {usage.output_tokens:,} トークン")
            print(f"  1 回の所要: 中央 {median:.1f} 秒 / 最大 {max(usage.latencies):.1f} 秒")
            print(f"  概算 ${cost:.2f} = ¥{cost * 154:,.0f}（為替 ¥154/$）")
        client.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
