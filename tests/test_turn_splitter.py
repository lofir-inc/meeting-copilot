"""チャンク内話者ターン分割の単体テスト。"""

from dataclasses import dataclass

import numpy as np

from src.audio.enrolled_diarizer import IdentifyResult
from src.audio.turn_splitter import TurnSplitConfig, split_turns


@dataclass
class FakeConfig:
    similarity_threshold: float = 0.75


@dataclass
class FakeDiarizer:
    partials: list[tuple[float, float, np.ndarray]]
    scores_by_embedding: dict[float, dict[str, float]]

    def __post_init__(self):
        self.config = FakeConfig()
        self.identify_calls: list[np.ndarray] = []

    def partial_embeddings(self, audio, sample_rate, rate):
        return self.partials

    def scores(self, embedding):
        nearest = min(self.scores_by_embedding, key=lambda value: abs(value - float(embedding[0])))
        return self.scores_by_embedding[nearest]

    def identify_embedding(self, embedding, duration_sec, at):
        self.identify_calls.append(embedding)
        scores = self.scores(embedding)
        name, similarity = max(scores.items(), key=lambda item: item[1])
        return IdentifyResult(name=name, similarity=similarity)

    @staticmethod
    def embed(audio, sample_rate):
        return np.array([1.0], dtype=np.float32)


def _partials(labels):
    return [
        (index * 0.4, (index + 1) * 0.4, np.array([float(index)], dtype=np.float32))
        for index, _ in enumerate(labels)
    ]


def _diarizer(labels):
    scores = {
        float(index): {label: 0.9, ("B" if label == "A" else "A"): 0.1}
        for index, label in enumerate(labels)
    }
    return FakeDiarizer(_partials(labels), scores)


def test_two_label_runs_become_two_turns():
    diarizer = _diarizer(["A", "A", "A", "B", "B"])
    turns = split_turns(np.ones(160000), 16000, diarizer, TurnSplitConfig(smooth=1), 0.0)
    assert [turn.speaker for turn in turns] == ["A", "B"]


def test_median_smoothing_removes_single_middle_label():
    diarizer = _diarizer(["A", "A", "B", "A", "A"])
    turns = split_turns(np.ones(160000), 16000, diarizer, TurnSplitConfig(smooth=3), 0.0)
    assert len(turns) == 1


def test_short_chunk_returns_one_turn():
    diarizer = _diarizer(["A", "A"])
    turns = split_turns(np.ones(16000), 16000, diarizer, TurnSplitConfig(), 0.0)
    assert len(turns) == 1


def test_short_run_is_absorbed_into_more_similar_neighbour():
    diarizer = _diarizer(["A", "A", "B", "A", "A"])
    turns = split_turns(
        np.ones(160000),
        16000,
        diarizer,
        TurnSplitConfig(smooth=1, min_turn_sec=0.8),
        0.0,
    )
    assert len(turns) == 1


def test_identify_embedding_is_called_once_per_run():
    diarizer = _diarizer(["A", "A", "A", "B", "B"])
    split_turns(np.ones(160000), 16000, diarizer, TurnSplitConfig(smooth=1), 0.0)
    assert len(diarizer.identify_calls) == 2


def test_short_unlabeled_run_is_absorbed_into_one_turn():
    labels = ["A", "A", None, "A", "A"]
    partials = _partials(labels)
    scores = {
        0.0: {"A": 0.9},
        1.0: {"A": 0.9},
        2.0: {"A": 0.1},
        3.0: {"A": 0.9},
        4.0: {"A": 0.9},
    }
    diarizer = FakeDiarizer(partials, scores)
    turns = split_turns(
        np.ones(160000),
        16000,
        diarizer,
        TurnSplitConfig(smooth=1, min_turn_sec=0.0, absorb_unlabeled_sec=3.0),
        0.0,
    )
    assert len(turns) == 1


def test_long_unlabeled_run_is_not_absorbed():
    labels = ["A", *([None] * 8), "B"]
    partials = _partials(labels)
    scores = {
        0.0: {"A": 0.9, "B": 0.1},
        9.0: {"B": 0.9, "A": 0.1},
    }
    scores.update({float(index): {"A": 0.1, "B": 0.1} for index in range(1, 9)})
    diarizer = FakeDiarizer(partials, scores)
    turns = split_turns(
        np.ones(160000),
        16000,
        diarizer,
        TurnSplitConfig(smooth=1, min_turn_sec=0.0, absorb_unlabeled_sec=3.0),
        0.0,
    )
    assert len(turns) == 3


def test_unlabeled_run_between_equally_similar_neighbours_is_kept():
    """吸収先が曖昧（両隣との cos の差 < absorb_margin）なら残す。"""
    labels = ["A", "A", None, "B", "B"]
    partials = _partials(labels)
    scores = {
        0.0: {"A": 0.9, "B": 0.1},
        1.0: {"A": 0.9, "B": 0.1},
        2.0: {"A": 0.1, "B": 0.1},
        3.0: {"B": 0.9, "A": 0.1},
        4.0: {"B": 0.9, "A": 0.1},
    }
    diarizer = FakeDiarizer(partials, scores)
    # 埋め込みは index を並べただけなので、None(2.0) と両隣(1.0 / 3.0) の cos は同じ
    turns = split_turns(
        np.ones(160000),
        16000,
        diarizer,
        TurnSplitConfig(smooth=1, min_turn_sec=0.0, absorb_unlabeled_sec=3.0, absorb_margin=0.10),
        0.0,
    )
    assert len(turns) == 3
