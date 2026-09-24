#!/usr/bin/env python3
"""録画ファイルを本番と同じパイプラインに流して試す（オフライン再生）。

Zoom / Google Meet のクラウド録画、MacWhisper 用に録った音声、過去のインタビュー録音など、
**手元にある会議の記録**を材料に、VAD → 話者判定 → STT → LLM要約 を通しで検証する。
実機・実会議を待たずに閾値を詰められる。

    python scripts/replay_meeting.py 録画.mp4 \\
        --enroll "自分=0:05-0:15" --enroll "田中=0:20-0:30" \\
        --session 2026-09-08-test

ここで検証できること／できないこと

  できる   : 話者判定の精度と閾値、Whisper の文字起こし品質、
             LLM の要約とタスク抽出、prompts/meeting_system.md の出来
  できない : 2系統キャプチャ・デバイス解決・音ずれ・起動セルフチェック
             （実機が要る。実会議前に1回は通しリハーサルをすること）

クラウド録画は「全員がミックスされた1本」である点に注意

  本番では自分の声はマイクから別系統で入るので**話者判定にかけない**。
  一方クラウド録画には自分の声も混ざっているので、**自分も登録する**必要がある。
  ＝ ここでの構成は本番と同じではない。話者判定に課される難易度は、
     むしろ本番より高い（本番は相手側だけを判定すればよい）。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.audio.enrolled_diarizer import DiarizerConfig, EnrolledDiarizer  # noqa: E402
from src.audio.turn_splitter import TurnSplitConfig, slice_chunk, split_turns  # noqa: E402
from src.audio.vad import AudioChunk, ChunkBuilder, VadConfig  # noqa: E402
from src.llm.ollama_client import LlmConfig, OllamaClient  # noqa: E402
from src.llm.state_updater import StateLoop, StateUpdater, StateUpdaterConfig  # noqa: E402
from src.output.markdown_writer import MarkdownWriter, OutputConfig  # noqa: E402
from src.stt.whisper_client import SttConfig, TranscriptSegment, WhisperClient  # noqa: E402
from src.ui.bus import EventBus  # noqa: E402
from src.ui.server import PortInUse, UiConfig, build_app, start_ui_server  # noqa: E402
from scripts.replay_state import load_prior_tasks  # noqa: E402

logger = logging.getLogger(__name__)


def _publish_replay_segment(bus: EventBus | None, state_loop: StateLoop | None, segment: TranscriptSegment) -> None:
    """リプレイ発話を UI とローリング状態更新へ同時に渡す。"""
    if bus is not None:
        bus.publish("segment", segment.to_dict())
    if state_loop is not None:
        state_loop.push(segment)


def _publish_replay_segment_with_participants(
    bus: EventBus | None,
    state_loop: StateLoop | None,
    actions: "ReplayUiActions",
    segment: TranscriptSegment,
) -> None:
    """リプレイ発話を配信し、参加者統計も更新する。"""
    # 改名後に「改名前のラベル」で届く発話（改名の瞬間に文字起こし中だったもの）を改名後の名前に揃える。
    #   揃えないと改名前後の話者が両方残る（運用者 の目視 2026-09-09 02:20）
    segment.speaker = actions.resolve_alias(segment.speaker)
    actions.record_segment(segment)
    _publish_replay_segment(bus, state_loop, segment)
    if bus is not None:
        bus.publish("participants", {"items": actions.participants()})


def decode_audio(path: Path, sample_rate: int, max_seconds: float | None = None) -> np.ndarray:
    """ffmpeg で任意の音声/動画をモノラル float32 に展開する。

    mp4/m4a/wav/mp3 など ffmpeg が読めるものは何でも通る。

    長さの制限は **ffmpeg 側で切る**（全体を展開してから捨てない）。
    48kHz float32 は 1 時間あたり約 690MB になるので、2 時間の録画を丸ごと
    メモリに載せると展開時に 2.7GB 級のピークが出る。
    """
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(path),
    ]
    if max_seconds:
        command += ["-t", str(max_seconds)]
    command += [
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    logger.info("ffmpeg で展開中: %s", path.name)
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg が失敗しました:\n{result.stderr.decode('utf-8', 'replace')[:2000]}"
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"音声トラックが取り出せませんでした: {path}")

    logger.info(
        "展開完了: %.1f 分 (%d サンプル @ %d Hz)",
        audio.size / sample_rate / 60, audio.size, sample_rate,
    )
    return audio.copy()


_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_timestamp(text: str) -> float:
    """"1:02:03" / "2:03" / "123" を秒に直す。"""
    match = _TIME_RE.match(text.strip())
    if not match:
        raise ValueError(f"時刻の書式が読めません: {text!r}（例: 1:02:03 / 2:03 / 123）")
    a, b, c = match.groups()
    parts = [p for p in (a, b, c) if p is not None]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_enroll_spec(spec: str) -> tuple[str, float, float]:
    """"自分=0:05-0:15" を (名前, 開始秒, 終了秒) に分解する。"""
    if "=" not in spec:
        raise ValueError(f"--enroll の書式が違います: {spec!r}（例: 自分=0:05-0:15）")
    name, _, span = spec.partition("=")
    if "-" not in span:
        raise ValueError(f"--enroll の区間が読めません: {spec!r}（例: 自分=0:05-0:15）")
    start_text, _, end_text = span.rpartition("-")
    start, end = parse_timestamp(start_text), parse_timestamp(end_text)
    if end <= start:
        raise ValueError(f"--enroll の終了が開始より前です: {spec!r}")
    return name.strip(), start, end


def build_diarizer(
    audio: np.ndarray,
    sample_rate: int,
    enroll_specs: list[str],
    speakers_path: Path | None,
    diarizer_cfg: DiarizerConfig,
) -> EnrolledDiarizer:
    """登録区間から話者を登録した diarizer を返す。"""
    if speakers_path and speakers_path.exists():
        diarizer = EnrolledDiarizer.load(
            speakers_path,
            similarity_threshold=diarizer_cfg.similarity_threshold,
            max_adapt_embeddings=diarizer_cfg.max_adapt_embeddings,
            adapt=diarizer_cfg.adapt,
        )
        logger.info("既存の話者登録を読み込み: %s (%d 名)", speakers_path, len(diarizer))
    else:
        diarizer = EnrolledDiarizer(diarizer_cfg)

    for spec in enroll_specs:
        name, start, end = parse_enroll_spec(spec)
        segment = audio[int(start * sample_rate):int(end * sample_rate)]
        diarizer.enroll(name, segment, sample_rate)
        print(f"  登録: {name}  ({start:.1f}s - {end:.1f}s / {end - start:.1f} 秒)")

    return diarizer


def enrolled_only(diarizer: EnrolledDiarizer) -> EnrolledDiarizer:
    """評価用に、名前付き登録時の embedding だけを残す。"""
    diarizer._speakers = {
        name: speaker
        for name, speaker in diarizer._speakers.items()
        if speaker.enrolled
    }
    for speaker in diarizer._speakers.values():
        speaker.adapt_embeddings.clear()
    return diarizer


def pipeline_version(root: Path) -> str:
    """現在のコミットの短縮 hash。git 外で実行されたときも結果を保存できる。"""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def replay(
    audio: np.ndarray,
    sample_rate: int,
    vad_cfg: VadConfig,
    diarizer: EnrolledDiarizer,
    whisper: WhisperClient,
    block_sec: float,
    turn_cfg: TurnSplitConfig,
    no_turn_split: bool = False,
    on_segment=None,
    realtime: bool = False,
) -> list[dict]:
    """音声を本番と同じ VAD → 話者判定 → STT に通す。"""
    builder = ChunkBuilder("replay", vad_cfg, sample_rate)
    block = int(block_sec * sample_rate)
    total_blocks = max(1, (audio.size + block - 1) // block)

    chunks: list[AudioChunk] = []
    for i in range(total_blocks):
        frame = audio[i * block:(i + 1) * block]
        if frame.size == 0:
            break
        chunks.extend(builder.feed(frame))

    # 末尾の言いさしを取りこぼさない
    tail = builder.feed(np.zeros(int(vad_cfg.silence_duration_sec * sample_rate) + block,
                                 dtype=np.float32))
    chunks.extend(tail)

    print(f"\n  VAD が {len(chunks)} 個の発話チャンクを切り出しました。文字起こし中…\n")

    results: list[dict] = []
    replay_started = time.monotonic()
    for chunk_index, chunk in enumerate(chunks, 1):
        if no_turn_split:
            embedding = diarizer.embed(chunk.audio, chunk.sample_rate)
            scores = diarizer.scores(embedding) if embedding is not None else {}
            outcome = diarizer.identify(chunk.audio, chunk.sample_rate, chunk.start_time)
            chunk.speaker = outcome.name
            turns = [(chunk, outcome.name, outcome.similarity, outcome.confident, outcome.is_new, scores, outcome.adapted)]
        else:
            turns = []
            for turn in split_turns(chunk.audio, chunk.sample_rate, diarizer, turn_cfg, chunk.start_time):
                turn_chunk = slice_chunk(chunk, turn)
                scores = diarizer.scores(turn.embedding) if turn.embedding.size else {}
                turns.append((turn_chunk, turn.speaker, turn.similarity, turn.confident, False, scores, turn.adapted))

        for turn_index, (turn_chunk, speaker, similarity, confident, is_new, scores, adapted) in enumerate(turns, 1):
            segment = whisper.transcribe(turn_chunk)
            text = segment.text if segment else ""
            diarizer.report_text(speaker, len(text))
            results.append({
                "index": len(results) + 1,
                "chunk_index": chunk_index,
                "turn_index": turn_index,
                "speaker": speaker,
                "similarity": round(similarity, 4),
                "confident": confident,
                "is_new": is_new,
                # この発話を centroid に取り込んだか（声紋接近 #10 の分析用）
                "adapted": adapted,
                "start_time": round(turn_chunk.start_time, 2),
                "end_time": round(turn_chunk.end_time, 2),
                "duration": round(turn_chunk.end_time - turn_chunk.start_time, 2),
                "text": text,
                # 1 位と 2 位の差。short_turn_margin を実測で詰める材料（2 位が無ければ None）
                "margin": _margin_of(scores),
                "scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
            })

            if realtime:
                delay = turn_chunk.start_time - (time.monotonic() - replay_started)
                if delay > 0:
                    time.sleep(delay)
            if on_segment is not None and text:
                on_segment(TranscriptSegment(
                    speaker=speaker,
                    text=text,
                    start_time=turn_chunk.start_time,
                    end_time=turn_chunk.end_time,
                    timestamp="",
                ))

            if text:
                mark = "  " if confident else "? "
                print(f"  {mark}[{_mmss(turn_chunk.start_time)}] {speaker} "
                      f"(sim={similarity:.3f}): {text[:70]}")

    return results


class ReplayUiActions:
    """リプレイ UI からの操作を diarizer へつなぐ。"""

    def __init__(self, diarizer: EnrolledDiarizer, session_dir: Path, config: dict | None = None) -> None:
        self.diarizer = diarizer
        self.session_dir = session_dir
        self.started_at = datetime.now().astimezone().isoformat()
        self.stats: dict[str, list[int]] = {}
        self._aliases: dict[str, str] = {}
        self._segment_chars: dict[float, int] = {}
        self._row_labels: dict[float, str] = {}
        llm_values = dict((config or {}).get("llm", {}))
        self._final_pass = str(llm_values.pop("final_pass", "none"))
        self._stt_cfg = SttConfig(**(config or {}).get("stt", {}))
        self._llm_cfg = LlmConfig(**llm_values)
        self._state_cfg = StateUpdaterConfig(**(config or {}).get("meeting", {}).get("state", {}))

    def record_segment(self, segment: TranscriptSegment) -> None:
        """配信した発話を参加者表示用に集計する。入口で改名の履歴を通す（呼び出し側任せにしない）。"""
        key = round(segment.start_time, 2)
        segment.speaker = self._row_labels.get(key, self.resolve_alias(segment.speaker))
        stats = self.stats.setdefault(segment.speaker, [0, 0])
        stats[0] += 1
        stats[1] += len(segment.text)
        self._segment_chars[key] = len(segment.text)

    def resolve_alias(self, name: str) -> str:
        """改名の連鎖（自分→MYSELF→自分）を辿って現在の表示名を返す。"""
        seen: set[str] = set()
        while name in self._aliases and name not in seen:
            seen.add(name)
            name = self._aliases[name]
        return name

    def rename_speaker(self, old: str, new: str) -> None:
        """リプレイ中の話者名を変更する。以後に届く旧ラベルの発話も新名に揃える。"""
        known = set(getattr(self.diarizer, "speaker_names", []) or [])
        if not known:
            known.update(getattr(self.diarizer, "enrolled_names", []) or [])
            known.update(getattr(self.diarizer, "unknown_names", []) or [])
        if old in known:
            try:
                self.diarizer.rename(old, new)
            except KeyError:
                logger.info("Replay diarizer speaker already absent during rename: %s", old)
        self._aliases[old] = new
        self._aliases.pop(new, None)   # 逆向きの改名で輪にならないように
        for key, name in self._row_labels.items():
            if name == old:
                self._row_labels[key] = new
        if old in self.stats:
            stats = self.stats.pop(old)
            target = self.stats.setdefault(new, [0, 0])
            target[0] += stats[0]
            target[1] += stats[1]

    def adjust_stats(self, old: str, new: str, chars: int) -> None:
        """1 発話の付け替えぶんだけ参加者統計を移す。"""
        if old in self.stats:
            self.stats[old][0] = max(0, self.stats[old][0] - 1)
            self.stats[old][1] = max(0, self.stats[old][1] - chars)
            if self.stats[old] == [0, 0] and old not in getattr(self.diarizer, "speaker_names", []) and old != "不明話者?":
                self.stats.pop(old)
        target = self.stats.setdefault(new, [0, 0])
        target[0] += 1
        target[1] += chars

    def relabel_segment(self, start_time: float, end_time: float, old: str, new: str) -> None:
        """一発話だけの話者訂正を記録する。"""
        output_cfg = OutputConfig(workspace_dir=str(self.session_dir))
        writer = MarkdownWriter(output_cfg)
        writer.append_correction({
            "at": datetime.now().astimezone().isoformat(),
            "start_time": start_time,
            "end_time": end_time,
            "old": old,
            "new": new,
        })
        self._row_labels[round(start_time, 2)] = new
        self.adjust_stats(old, new, self._segment_chars.get(round(start_time, 2), 0))
        logger.info("Replay segment relabeled from %s to %s at %.3f", old, new, start_time)

    def session_info(self) -> dict:
        """UI の初期表示用情報を返す。"""
        return {
            "session_name": self.session_dir.name,
            "started_at": self.started_at,
            "self_name": "",
            "enrolled": self.diarizer.enrolled_names,
            "models": {
                "stt": {"name": self._stt_cfg.model, "where": "ローカル（mlx）"},
                "summary": {"name": self._llm_cfg.model, "where": "ローカル（Ollama）", "think": self._state_cfg.think},
                "final_pass": {"name": "Claude CLI", "where": "外部（テキストのみ・音声は出ない）"} if self._final_pass != "none" else None,
            },
        }

    def participants(self) -> list[dict]:
        """登録済み・検出済み話者と発話統計を返す。"""
        enrolled = self.diarizer.enrolled_names
        unknown = self.diarizer.unknown_names
        names = list(dict.fromkeys([*enrolled, *unknown, *self.stats]))
        known = set(getattr(self.diarizer, "speaker_names", []) or [])
        items = [
            {"name": name, "enrolled": name in enrolled, "utterances": self.stats.get(name, [0, 0])[0], "chars": self.stats.get(name, [0, 0])[1]}
            for name in names
            if name != "不明話者?" and (self.stats.get(name, [0, 0]) != [0, 0] or name in known)
        ]
        items.append({"name": "不明話者?", "enrolled": False, "utterances": self.stats.get("不明話者?", [0, 0])[0], "chars": self.stats.get("不明話者?", [0, 0])[1]})
        return items


def relabel_dissolved(results: list[dict], dissolved: list[str]) -> int:
    """解体された不明話者の発話を `不明話者?` に付け直す。付け直した件数を返す。

    昇格後に「混合」と判明したクラスタは、それまでの発話にも人の名前を残さない
    （間違った名前は名前が無いより悪い）。本番では Phase 3 の UI が同じ付け直しを行う。
    """
    if not dissolved:
        return 0
    targets = set(dissolved)
    count = 0
    for row in results:
        if row["speaker"] in targets:
            row["speaker"] = "不明話者?"
            row["confident"] = False
            count += 1
    return count


def _py(value):
    """JSON へ出す直前に numpy スカラー／配列を Python 型へ落とす関門。

    numpy 型の漏れはこれで 3 件目（PendingCluster の __eq__、adapted の numpy.bool、…）。
    `json.dumps(default=)` は「直列化できない型に出会ったとき」しか呼ばれず、numpy float のように
    素通りする型を取りこぼすので、事前変換にする（aa の指摘 2026-09-08 17:10）。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(item) for item in value]
    return value


def _margin_of(scores: dict[str, float]) -> float | None:
    """類似度の 1 位と 2 位の差。話者が 2 名未満なら None。"""
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return None
    return round(ordered[0] - ordered[1], 4)


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _pair_similarity_json(values: dict[tuple[str, str], float]) -> dict[str, float]:
    """話者ペアを JSON で安定して扱えるキーに変換する。"""
    return {"|".join(pair): value for pair, value in values.items()}


def report(
    results: list[dict],
    threshold: float,
    pair_similarity_enrolled: dict[tuple[str, str], float],
    pair_similarity_final: dict[tuple[str, str], float],
    unknown_vs_enrolled: dict[str, dict[str, float]],
) -> None:
    """話者ごとの分布を出す。閾値を決める材料になる。"""
    print()
    print("=" * 70)
    print("  話者別の集計")
    print("=" * 70)

    by_speaker: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_speaker[r["speaker"]].append(r["similarity"])

    counts = Counter(r["speaker"] for r in results)
    for speaker, count in counts.most_common():
        sims = [s for s in by_speaker[speaker] if s >= 0]
        if sims:
            print(f"  {speaker:14s} {count:4d} 発話   "
                  f"類似度 min={min(sims):.3f} 中央={float(np.median(sims)):.3f} max={max(sims):.3f}")
        else:
            print(f"  {speaker:14s} {count:4d} 発話")

    low = [r for r in results if r["confident"] and r["similarity"] < threshold + 0.05]
    unknown = [r for r in results if not r["confident"]]

    print()
    print(f"  閾値 {threshold:.2f} で判定できなかった発話: {len(unknown)} / {len(results)}")
    if low:
        print(f"  閾値ぎりぎり（+0.05 以内）で通った発話  : {len(low)} 件 ← 閾値を上げると不明話者に落ちる")
    print()
    print("  閾値の詰め方: --threshold を変えて再実行し、"
          "「不明話者N」が減りすぎて別人が混ざらない値を探す。")
    print("    replay_result.json の scores を見ると、誰と誰が紛れているか分かる。")
    for pair, enrolled in pair_similarity_enrolled.items():
        final = pair_similarity_final.get(pair, enrolled)
        print(f"  声紋の接近: {pair[0]} vs {pair[1]} 登録時 {enrolled:.3f} → 終了時 {final:.3f}")
    for unknown_name, similarities in unknown_vs_enrolled.items():
        values = " / ".join(
            f"vs {name} {similarity:.2f}"
            for name, similarity in similarities.items()
        )
        suspicion = " ← 0.75 超なら断片の疑い" if any(
            similarity > 0.75 for similarity in similarities.values()
        ) else ""
        print(f"  {unknown_name}: {values}{suspicion}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="録画ファイルを本番と同じパイプラインに流して試す",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("media", help="音声/動画ファイル（mp4・m4a・wav・mp3 …）")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--session", default=None, help="セッション名（既定はファイル名）")
    parser.add_argument(
        "--enroll", action="append", default=[], metavar="名前=開始-終了",
        help='話者の登録区間。複数指定可。例: --enroll "自分=0:05-0:15"',
    )
    parser.add_argument("--speakers", default=None, help="既存の speakers.json を読み込む")
    parser.add_argument(
        "--enrolled-only", action="store_true",
        help="--speakers の登録済み話者の enroll_embeddings だけを評価に使う",
    )
    parser.add_argument("--threshold", type=float, default=None, help="類似度の閾値を上書き")
    parser.add_argument("--vad", choices=["rms", "silero"], default=None, help="VAD バックエンドを上書き")
    parser.add_argument("--no-turn-split", action="store_true", help="旧来のチャンク単位で話者判定する")
    parser.add_argument(
        "--diarizer-opt", action="append", default=[], metavar="キー=値",
        help="DiarizerConfig の任意フィールドを上書き（例: --diarizer-opt mixture_similarity=0.70）。"
             "settings を書き換えずに閾値の実験をするため（read-only の検査セッションが使う）",
    )
    parser.add_argument(
        "--max-chunk-sec", type=float, default=None,
        help="1チャンクの最大長を上書き。テンポの速い会話では短くする"
             "（長いと1チャンクに複数人が入り、embedding が混ざって判定できなくなる）",
    )
    parser.add_argument(
        "--silence-sec", type=float, default=None,
        help="発話を切る無音の長さを上書き。話者交代の短い間を拾うには小さくする",
    )
    parser.add_argument(
        "--min-chunk-sec", type=float, default=None, help="これより短いチャンクは捨てる",
    )
    parser.add_argument("--max-minutes", type=float, default=None, help="先頭N分だけ処理する")
    parser.add_argument("--no-llm", action="store_true", help="要約を飛ばし、文字起こしまでで止める")
    parser.add_argument("--serve", action="store_true", help="ローカル Web UI へ発話と状態を配信する")
    parser.add_argument("--port", type=int, default=8765, help="--serve 時の UI ポート")
    parser.add_argument("--realtime", action="store_true", help="--serve の発話を会議時刻どおりに流す")
    parser.add_argument("--prep-dir", type=Path, default=None, help="状態更新へ渡す事前資料ディレクトリ")
    parser.add_argument(
        "--no-adapt", action="store_true",
        help="実行中に embedding を取り込まない（1回の誤判定が後続を汚すのを防ぐ）",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    sample_rate = config.get("audio", {}).get("sample_rate", 48000)
    meeting = config.get("meeting", {})
    vad_values = dict(config.get("vad", {}))
    vad_values.update(meeting.get("vad", {}))
    vad_cfg = VadConfig(**vad_values)
    if args.vad is not None:
        vad_cfg.backend = args.vad
    if args.max_chunk_sec is not None:
        vad_cfg.max_chunk_sec = args.max_chunk_sec
    if args.silence_sec is not None:
        vad_cfg.silence_duration_sec = args.silence_sec
    if args.min_chunk_sec is not None:
        vad_cfg.min_chunk_sec = args.min_chunk_sec
    diarizer_values = dict(meeting.get("diarizer", {}))
    diarizer_values.setdefault("similarity_threshold", meeting.get("similarity_threshold", 0.75))
    diarizer_values.setdefault("max_adapt_embeddings", meeting.get("max_adapt_embeddings", 10))
    if args.threshold is not None:
        diarizer_values["similarity_threshold"] = args.threshold
    if args.no_adapt:
        diarizer_values["adapt"] = False
    for option in args.diarizer_opt:
        key, _, raw = option.partition("=")
        if key not in DiarizerConfig.__dataclass_fields__:
            parser.error(f"--diarizer-opt: 未知のフィールド {key!r}")
        field_type = type(getattr(DiarizerConfig(), key))
        diarizer_values[key] = raw.lower() in {"1", "true", "yes"} if field_type is bool else field_type(raw)
    diarizer_cfg = DiarizerConfig(**diarizer_values)
    turn_cfg = TurnSplitConfig(**meeting.get("turn_split", {}))
    threshold = diarizer_cfg.similarity_threshold

    media = Path(args.media).expanduser()
    session_name = args.session or f"replay-{media.stem}"
    session_dir = root / "workspace" / "sessions" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  録画リプレイ（オフライン検証）")
    print("=" * 70)
    print(f"  入力     : {media}")
    print(f"  セッション: {session_dir}")
    print(f"  閾値     : {threshold}")
    print(f"  VAD      : 最大 {vad_cfg.max_chunk_sec}s / 無音 {vad_cfg.silence_duration_sec}s "
          f"/ 最小 {vad_cfg.min_chunk_sec}s")
    print()
    print("  クラウド録画は全員がミックスされた1本です。本番と違い自分の声も")
    print("    判定対象になるので、--enroll には**自分も**登録してください。")
    print()

    audio = decode_audio(
        media, sample_rate,
        max_seconds=args.max_minutes * 60 if args.max_minutes else None,
    )
    if args.max_minutes:
        print(f"  先頭 {args.max_minutes} 分だけ読み込みました。\n")

    speakers_path = Path(args.speakers) if args.speakers else session_dir / "speakers.json"
    diarizer = build_diarizer(audio, sample_rate, args.enroll, speakers_path, diarizer_cfg)
    if args.enrolled_only:
        diarizer = enrolled_only(diarizer)
        print("  （--enrolled-only: 登録済み話者の enroll_embeddings だけを使用）\n")
    if args.no_adapt:
        print("  （--no-adapt: 実行中の embedding 取り込みを止めています）\n")
    if not len(diarizer):
        print("  ⚠ 誰も登録されていません。全員が「不明話者N」になります。")
        print("    それでも「何人いるか」「発話がどう分かれるか」は分かります。\n")

    bus: EventBus | None = None
    ollama: OllamaClient | None = None
    state_loop: StateLoop | None = None
    writer: MarkdownWriter | None = None
    ui_actions: ReplayUiActions | None = None
    if args.serve:
        # 同名セッションの前回の state.json が残っていると、UI が開いた瞬間に古い右カラムを表示する。
        #   リプレイは毎回ゼロから流すので、古い状態は消してから始める
        for stale in ("state.json", "state.json.tmp"):
            (session_dir / stale).unlink(missing_ok=True)
        bus = EventBus()
        ui_cfg = UiConfig(host="127.0.0.1", port=args.port, open_browser=False)
        try:
            ui_actions = ReplayUiActions(diarizer, session_dir, config)
            start_ui_server(build_app(bus, ui_actions, session_dir), ui_cfg)
            print(f"  UI        : http://{ui_cfg.host}:{ui_cfg.port}/")
        except PortInUse as exc:
            # 黙って続けると、ブラウザは前のセッションを見続ける。ここで止める
            print(f"\n  ✗ {exc}\n", file=sys.stderr)
            return 2
        except Exception:
            logger.warning("ローカル UI の起動に失敗しました", exc_info=True)
        bus.publish("status", {"phase": "replay", "warnings": [], "note": "リプレイ再生中。マイクとレベル計は無効"})
        if not args.no_llm:
            ollama = OllamaClient(LlmConfig(**config.get("llm", {})))
            if ollama.health_check():
                output_cfg = OutputConfig(workspace_dir=str(session_dir))
                output_cfg.title = "会議（リプレイ）"
                output_cfg.items_label = "前回タスク 消化状況"
                writer = MarkdownWriter(output_cfg, interview_purpose=f"リプレイ検証: {media.name}")
                writer.init_files("（リプレイ検証のため事前資料なし）")
                state_prompt = (root / "prompts" / "meeting_state_system.md").read_text(encoding="utf-8").strip()
                updater = StateUpdater(ollama, state_prompt, load_prior_tasks(args.prep_dir), StateUpdaterConfig())

                def on_state(state, stats, changes) -> None:
                    """状態更新を UI とセッションファイルへ反映する。"""
                    del changes
                    bus.publish("state", {"state": state.to_dict(), "latency_sec": stats.latency_sec, "parse_ok": stats.parse_ok})
                    writer.write_state(state)
                    writer.write_state_json(state)

                state_loop = StateLoop(updater, on_state=on_state)
                state_loop.start()
            else:
                print("\n  ⚠ Ollama に接続できないので右カラムは更新されません（文字起こしは配信します）。")
                bus.publish("status", {"phase": "replay", "warnings": ["右カラムは更新されません（Ollama 不通）"], "note": "リプレイ再生中。マイクとレベル計は無効"})
                ollama.close()
                ollama = None

    whisper = WhisperClient(SttConfig(**config.get("stt", {})))
    logger.info("Whisper モデルをロード中…")
    whisper._ensure_model()

    pair_similarity_enrolled = diarizer.pairwise_similarity()
    started = time.perf_counter()
    results = replay(
        audio,
        sample_rate,
        vad_cfg,
        diarizer,
        whisper,
        config.get("audio", {}).get("chunk_duration_sec", 0.1),
        turn_cfg,
        args.no_turn_split,
        on_segment=lambda segment: _publish_replay_segment(bus, state_loop, segment) if ui_actions is None else _publish_replay_segment_with_participants(bus, state_loop, ui_actions, segment),
        realtime=args.realtime,
    )
    elapsed_sec = time.perf_counter() - started
    if state_loop is not None:
        state_loop.stop()
        final_state = state_loop.flush(60)
        if writer is not None:
            writer.write_state(final_state)
            writer.write_state_json(final_state)
    pair_similarity_final = diarizer.pairwise_similarity()
    unknown_vs_enrolled = diarizer.unknown_vs_enrolled()
    relabeled = relabel_dissolved(results, diarizer.dissolved_names)
    if relabeled:
        print(f"\n  混合と判明して解体した不明話者: {diarizer.dissolved_names}（{relabeled} 発話を 不明話者? に付け直し）")

    # --- 保存 ---
    (session_dir / "replay_result.json").write_text(
        json.dumps(_py({
            "media": str(media),
            "elapsed_sec": round(elapsed_sec, 3),
            "vad": asdict(vad_cfg),
            "threshold": threshold,
            "pipeline_version": pipeline_version(root),
            "pair_similarity_enrolled": _pair_similarity_json(pair_similarity_enrolled),
            "pair_similarity_final": _pair_similarity_json(pair_similarity_final),
            "self_drift": diarizer.self_drift(),
            "mixture_rejections": diarizer.mixture_rejections,
            "dissolved": list(diarizer.dissolved_names),
            "unknown_vs_enrolled": unknown_vs_enrolled,
            "turn_split": asdict(turn_cfg),
            "segments": results,
        }), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(session_dir / "transcripts.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            if r["text"]:
                f.write(json.dumps({
                    "speaker": r["speaker"], "text": r["text"],
                    "start_time": r["start_time"], "end_time": r["end_time"],
                    "timestamp": "",
                }, ensure_ascii=False) + "\n")
    diarizer.save(session_dir / "speakers.json")

    report(
        results,
        threshold,
        pair_similarity_enrolled,
        pair_similarity_final,
        unknown_vs_enrolled,
    )

    if args.no_llm:
        print(f"\n  文字起こしまで完了: {session_dir}")
        if args.serve:
            print("Ctrl+C で終了")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0
        return 0

    # --- LLM 要約 ---
    transcript = "\n".join(
        f"[{_mmss(r['start_time'])}] {r['speaker']}: {r['text']}"
        for r in results if r["text"]
    )
    if not transcript:
        print("\n  ⚠ 文字起こしが空だったので要約は行いません。")
        return 1

    if args.serve:
        # サーバを生かしたまま待つ（右カラムの最終状態を見られるように）。Ctrl+C で終了
        if ollama is not None:
            ollama.close()
        print(f"\n  完了: {session_dir}")
        print("  UI を開いたままにしています。Ctrl+C で終了")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return 0

    ollama = OllamaClient(LlmConfig(**config.get("llm", {})))
    if not ollama.health_check():
        print("\n  ⚠ Ollama に接続できないので要約を飛ばします（文字起こしは保存済み）。")
        return 1

    if not args.serve:
        prompt_path = root / "prompts" / "meeting_system.md"
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        print("\n  要約中…")
        response = ollama.summarize(
            system_prompt=system_prompt,
            hearing_items="（リプレイ検証のため事前資料なし）",
            overall_summary_so_far="",
            recent_transcript=transcript,
        )

    if not args.serve:
        output_cfg = OutputConfig(workspace_dir=str(session_dir))
        output_cfg.title = "会議（リプレイ）"
        output_cfg.items_label = "前回タスク 消化状況"
        writer = MarkdownWriter(output_cfg, interview_purpose=f"リプレイ検証: {media.name}")
        writer.init_files("（リプレイ検証のため事前資料なし）")
        writer.overwrite_summary(response)
        writer.append_log(response.chunk_summary)
    ollama.close()

    print()
    print("=" * 70)
    print("  完了")
    print("=" * 70)
    print(f"  要約       : {session_dir / 'interview_summary.md'}")
    print(f"  時系列ログ : {session_dir / 'interview_log.md'}")
    print(f"  文字起こし : {session_dir / 'transcripts.jsonl'}")
    print(f"  判定の詳細 : {session_dir / 'replay_result.json'}  ← 閾値を詰めるならここ")
    if args.serve:
        print("Ctrl+C で終了")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
