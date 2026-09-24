import pytest

from scripts.eval_replay import metrics


def test_metrics_fix_definitions_for_synthetic_segments():
    result = metrics([
        {"speaker": "A", "text": "abcdefghij", "duration": 9.9, "similarity": 0.9, "confident": True},
        {"speaker": "不明話者1", "text": "x" * 20, "duration": 10.0, "similarity": 0.4, "confident": False},
        {"speaker": "不明話者?", "text": "", "duration": 1.0, "similarity": -1.0, "confident": False},
    ], 10.0)

    assert result["segments"] == 3
    assert result["total_chars"] == 30
    assert result["empty_text_rate"] == pytest.approx(1 / 3)
    assert result["named_char_rate"] == pytest.approx(1 / 3)
    assert result["unknown_n_char_rate"] == pytest.approx(2 / 3)
    assert result["unknown_q_char_rate"] == 0.0
    assert result["cap_hit_rate"] == pytest.approx(2 / 3)
    assert result["unknown_clusters"] == {"不明話者1": {"utterances": 1, "chars": 20}}
    assert result["largest_unknown_share"] == 1.0
    assert result["unknown_clusters_ge"] == {5: 0, 20: 0}
    assert result["confident_sim_median"] == 0.9


def test_cap_hit_rate_counts_chunks_not_turns_when_chunk_index_is_present():
    """ターン分割後の JSON では、チャンク単位で上限打ち切りを数える。"""
    segments = [
        # チャンク 1 = 4.0 + 6.0 = 10.0 秒（上限打ち切り）
        {"speaker": "A", "text": "a" * 10, "duration": 4.0, "chunk_index": 1, "confident": True, "similarity": 0.9},
        {"speaker": "B", "text": "b" * 10, "duration": 6.0, "chunk_index": 1, "confident": True, "similarity": 0.9},
        # チャンク 2 = 3.0 秒
        {"speaker": "A", "text": "c" * 10, "duration": 3.0, "chunk_index": 2, "confident": True, "similarity": 0.9},
    ]
    result = metrics(segments, 10.0)
    assert result["cap_hit_rate"] == pytest.approx(1 / 2)
