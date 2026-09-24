"""VAD チャンクを部分 embedding で話者ターンへ分割する。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.audio.enrolled_diarizer import EnrolledDiarizer, IdentifyResult, _cosine
from src.audio.vad import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class TurnSplitConfig:
    enabled: bool = True
    rate: float = 2.5
    min_turn_sec: float = 0.8
    smooth: int = 3
    absorb_unlabeled_sec: float = 3.0
    absorb_margin: float = 0.10
    """両隣がラベル付きのとき、吸収先を決めるのに必要な cos の差。差が小さければ吸収しない。"""


@dataclass
class Turn:
    """チャンク先頭からの相対時刻で表す話者ターン。"""

    start: float
    end: float
    speaker: str
    similarity: float
    confident: bool
    embedding: np.ndarray
    adapted: bool = False


@dataclass
class _Run:
    start: float
    end: float
    embeddings: list[np.ndarray]
    label: str | None

    @property
    def embedding(self) -> np.ndarray:
        return np.mean(self.embeddings, axis=0)


def split_turns(
    audio: np.ndarray,
    sample_rate: int,
    diarizer: EnrolledDiarizer,
    cfg: TurnSplitConfig,
    at: float,
) -> list[Turn]:
    """音声チャンクを平滑化した部分 embedding の連続区間へ分割する。"""
    duration = np.asarray(audio).size / sample_rate
    if not cfg.enabled or duration < 2.0:
        return [_single_turn(audio, sample_rate, diarizer, duration, at)]
    partials = diarizer.partial_embeddings(audio, sample_rate, cfg.rate)
    if len(partials) <= 1:
        return [_single_turn(audio, sample_rate, diarizer, duration, at)]
    labels = [_best_label(diarizer, embedding) for _, _, embedding in partials]
    labels = _smooth_labels(labels, cfg.smooth)
    runs = _make_runs(partials, labels)
    runs = _absorb_short_runs(runs, cfg.min_turn_sec)
    runs = _absorb_unlabeled_runs(runs, cfg.absorb_unlabeled_sec, cfg.absorb_margin)
    logger.debug("チャンク内ターンを分割: partials=%d runs=%d", len(partials), len(runs))
    return [_identify_run(run, diarizer, at) for run in runs]


def slice_chunk(chunk: AudioChunk, turn: Turn) -> AudioChunk:
    """ターン範囲の音声を切り出して時刻と話者を反映する。"""
    start = int(round(turn.start * chunk.sample_rate))
    end = int(round(turn.end * chunk.sample_rate))
    return AudioChunk(
        speaker=turn.speaker,
        audio=chunk.audio[start:end],
        start_time=chunk.start_time + turn.start,
        end_time=chunk.start_time + turn.end,
        sample_rate=chunk.sample_rate,
    )


def _single_turn(
    audio: np.ndarray,
    sample_rate: int,
    diarizer: EnrolledDiarizer,
    duration: float,
    at: float,
) -> Turn:
    embedding = diarizer.embed(audio, sample_rate)
    if embedding is None:
        result = IdentifyResult(name="不明話者?", similarity=-1.0, confident=False)
        embedding = np.empty(0, dtype=np.float32)
    else:
        result = diarizer.identify_embedding(embedding, duration, at)
    return Turn(
        0.0,
        duration,
        result.name,
        result.similarity,
        result.confident,
        embedding,
        adapted=getattr(result, "adapted", False),
    )


def _best_label(diarizer: EnrolledDiarizer, embedding: np.ndarray) -> str | None:
    """候補プールを変えずに、窓の最良既知ラベルだけを決める。"""
    scores = diarizer.scores(embedding)
    if not scores:
        return None
    name, similarity = max(scores.items(), key=lambda item: item[1])
    return name if similarity >= diarizer.config.similarity_threshold else None


def _smooth_labels(labels: list[str | None], width: int) -> list[str | None]:
    """奇数幅の多数決で窓ラベルを平滑化する。"""
    if width <= 1 or len(labels) < 2:
        return labels
    radius = width // 2
    smoothed: list[str | None] = []
    for index, label in enumerate(labels):
        window = labels[max(0, index - radius) : min(len(labels), index + radius + 1)]
        counts = {candidate: window.count(candidate) for candidate in set(window)}
        best_count = max(counts.values())
        choices = [candidate for candidate, count in counts.items() if count == best_count]
        smoothed.append(label if label in choices else choices[0])
    return smoothed


def _make_runs(
    partials: list[tuple[float, float, np.ndarray]],
    labels: list[str | None],
) -> list[_Run]:
    """連続する同ラベルの部分窓を一つの run にまとめる。"""
    runs: list[_Run] = []
    for (start, end, embedding), label in zip(partials, labels):
        if not runs or runs[-1].label != label:
            run = _Run(start=start, end=end, embeddings=[embedding], label=label)
            runs.append(run)
        else:
            runs[-1].end = end
            runs[-1].embeddings.append(embedding)
    return runs


def _absorb_short_runs(runs: list[_Run], minimum: float) -> list[_Run]:
    """短い run を embedding がより似た隣接 run へ吸収する。"""
    index = 0
    while len(runs) > 1 and index < len(runs):
        run = runs[index]
        if run.end - run.start + 1e-9 >= minimum:
            index += 1
            continue
        if index == 0:
            target = 1
        elif index == len(runs) - 1:
            target = index - 1
        else:
            left = _cosine(run.embedding, runs[index - 1].embedding)
            right = _cosine(run.embedding, runs[index + 1].embedding)
            target = index - 1 if left >= right else index + 1
        receiver = runs[target]
        receiver.start = min(receiver.start, run.start)
        receiver.end = max(receiver.end, run.end)
        receiver.embeddings.extend(run.embeddings)
        runs.pop(index)
        runs = _coalesce_runs(runs)
        index = 0
    return runs


def _coalesce_runs(runs: list[_Run]) -> list[_Run]:
    """隣接する同ラベル run を、短い区間の吸収後に再併合する。"""
    coalesced: list[_Run] = []
    for run in runs:
        if coalesced and coalesced[-1].label == run.label:
            coalesced[-1].end = run.end
            coalesced[-1].embeddings.extend(run.embeddings)
        else:
            coalesced.append(run)
    return coalesced


def _absorb_unlabeled_runs(runs: list[_Run], minimum: float, margin: float) -> list[_Run]:
    """短い未ラベル run を最も似たラベル付き隣接 run へ吸収する。

    吸収先を間違えると誤りが伝播するので、両隣がラベル付きのときは
    cos の差が margin 以上あるときだけ吸収する。片側しか無ければ吸収する。
    """
    index = 0
    while index < len(runs):
        run = runs[index]
        if run.label is not None or run.end - run.start + 1e-9 >= minimum:
            index += 1
            continue
        candidates = [
            neighbor
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(runs) and runs[neighbor].label is not None
        ]
        if not candidates:
            index += 1
            continue
        similarities = sorted(
            ((_cosine(run.embedding, runs[neighbor].embedding), neighbor) for neighbor in candidates),
            reverse=True,
        )
        same_label = len(candidates) > 1 and runs[candidates[0]].label == runs[candidates[1]].label
        # 両隣が同じ話者なら吸収先の間違いは起きないので、差は問わない
        if (
            not same_label
            and len(similarities) > 1
            and similarities[0][0] - similarities[1][0] < margin
        ):
            logger.debug("未ラベル run の吸収先が曖昧なので残す (差=%.3f)", similarities[0][0] - similarities[1][0])
            index += 1
            continue
        target = similarities[0][1]
        receiver = runs[target]
        receiver.start = min(receiver.start, run.start)
        receiver.end = max(receiver.end, run.end)
        receiver.embeddings.extend(run.embeddings)
        runs.pop(index)
        runs = _coalesce_runs(runs)
        index = 0
    return runs


def _identify_run(run: _Run, diarizer: EnrolledDiarizer, at: float) -> Turn:
    """run の平均 embedding を一度だけ話者照合する。"""
    embedding = run.embedding
    result = diarizer.identify_embedding(embedding, run.end - run.start, at + run.start)
    return Turn(
        run.start,
        run.end,
        result.name,
        result.similarity,
        result.confident,
        embedding,
        adapted=result.adapted,
    )
