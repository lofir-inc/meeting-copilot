"""文字起こしの採点スクリプトのテスト。"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from macwhisper_reference import find_offset  # noqa: E402
from score_transcript import Row, edit_distance, load, score  # noqa: E402

SPAN = (0.0, 60.0)


def _rows(*items: tuple[float, str, str]) -> list[Row]:
    return [Row(start, start + 2.0, speaker, text) for start, speaker, text in items]


def test_edit_distance_counts_character_changes():
    assert edit_distance("あいう", "あいう") == 0
    assert edit_distance("あいう", "あえう") == 1
    assert edit_distance("", "あい") == 2


def test_load_applies_the_shift(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"start_time": 1.0, "end_time": 2.0, "speaker": "自分", "text": "はい"}) + "\n", encoding="utf-8")
    assert load(path, shift=71.3)[0].start == 72.3


def test_identical_transcripts_score_zero():
    rows = _rows((0.0, "自分", "台湾のサブドメインをつけましょう"), (10.0, "参加者A", "それでいいと思います"))
    result = score(rows, rows, SPAN)
    assert result["cer"] == 0.0
    assert result["missed_lines"] == 0
    assert result["extra_lines"] == 0


def test_missed_and_extra_lines_are_counted():
    reference = _rows((0.0, "自分", "台湾のサブドメインをつけましょう"), (20.0, "参加者A", "在庫の話に戻ります"))
    ours = _rows((0.0, "自分", "台湾のサブドメインをつけましょう"), (40.0, "不明話者?", "impressive"))
    result = score(ours, reference, SPAN)
    assert result["missed_lines"] == 1
    assert result["extra_lines"] == 1
    assert "impressive" in result["extra_examples"][0]


def test_speaker_mapping_follows_the_majority():
    """こちらの「不明話者1」が正解の「参加者B」に対応していれば、その対応で一致とみなす。"""
    reference = _rows(
        (0.0, "参加者B", "まだそこらへん固まってないですもんね"),
        (10.0, "参加者B", "てかもう納品物みたいな感じですね"),
        (20.0, "自社自分", "こちらで問い合わせフォームを作れます"),
    )
    ours = _rows(
        (0.0, "不明話者1", "まだそこらへん固まってないですもんね"),
        (10.0, "不明話者1", "てかもう納品物みたいな感じですね"),
        (20.0, "自分", "こちらで問い合わせフォームを作れます"),
    )
    result = score(ours, reference, SPAN)
    assert result["speaker_mapping"] == {"不明話者1": "参加者B", "自分": "自社自分"}
    assert result["speaker_correct"] == result["speaker_matched"] == 3


def test_wrong_speaker_is_not_counted_as_correct():
    reference = _rows((0.0, "参加者B", "まだそこらへん固まってないですもんね"), (10.0, "参加者A", "在庫の話に戻りますけれども"))
    ours = _rows((0.0, "参加者B", "まだそこらへん固まってないですもんね"), (10.0, "参加者B", "在庫の話に戻りますけれども"))
    result = score(ours, reference, SPAN)
    assert result["speaker_matched"] == 2
    assert result["speaker_correct"] == 1


def test_find_offset_locates_our_recording_inside_the_reference():
    """時刻合わせ: こちらの録音が基準の何秒目から始まるかを相互相関で当てる。"""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal(16000 * 40).astype(np.float32) * 0.1
    ours = reference[16000 * 7 :].copy()
    offset, correlation = find_offset(ours, reference, probes=3, window_sec=2.0)
    assert offset == 7.0
    assert correlation > 0.99
