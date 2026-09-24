"""録画リプレイツールの引数解釈のテスト。

音声処理そのものは本番と同じクラスを使うので、ここでは
「時刻・登録区間の書式を取り違えないか」だけを見る。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "replay_meeting.py"
_spec = importlib.util.spec_from_file_location("replay_meeting", _SCRIPT)
replay_meeting = importlib.util.module_from_spec(_spec)
sys.modules["replay_meeting"] = _spec.name and replay_meeting
_spec.loader.exec_module(replay_meeting)

parse_timestamp = replay_meeting.parse_timestamp
parse_enroll_spec = replay_meeting.parse_enroll_spec


def test_replay_ui_actions_record_segment_correction(tmp_path: Path) -> None:
    """リプレイ UI の行単位訂正は corrections.jsonl に追記する。"""
    class FakeDiarizer:
        enrolled_names: list[str] = []

        def rename(self, old: str, new: str) -> None:
            """テストでは話者名変更を何もしない。"""
            del old, new

    actions = replay_meeting.ReplayUiActions(FakeDiarizer(), tmp_path)
    actions.relabel_segment(1.0, 2.0, "不明話者?", "田中")
    records = (tmp_path / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert '"new": "田中"' in records[0]


class TestParseTimestamp:
    def test_seconds_only(self):
        assert parse_timestamp("123") == 123.0

    def test_minutes_and_seconds(self):
        assert parse_timestamp("2:03") == 123.0

    def test_hours(self):
        assert parse_timestamp("1:02:03") == 3723.0

    def test_fractional_seconds(self):
        assert parse_timestamp("10.5") == pytest.approx(10.5)
        assert parse_timestamp("0:10.5") == pytest.approx(10.5)

    def test_surrounding_space_is_ignored(self):
        assert parse_timestamp(" 2:03 ") == 123.0

    @pytest.mark.parametrize("bad", ["", "abc", "1:2:3:4", "1;30"])
    def test_rejects_garbage(self, bad):
        with pytest.raises(ValueError):
            parse_timestamp(bad)


class TestParseEnrollSpec:
    def test_basic(self):
        assert parse_enroll_spec("自分=0:05-0:15") == ("自分", 5.0, 15.0)

    def test_plain_seconds(self):
        assert parse_enroll_spec("田中=10.2-16.0") == ("田中", pytest.approx(10.2), pytest.approx(16.0))

    def test_hour_long_recording(self):
        name, start, end = parse_enroll_spec("佐藤=1:02:03-1:02:30")
        assert (name, start, end) == ("佐藤", 3723.0, 3750.0)

    def test_name_is_stripped(self):
        assert parse_enroll_spec("  自分  =0:05-0:15")[0] == "自分"

    def test_rejects_missing_equals(self):
        with pytest.raises(ValueError):
            parse_enroll_spec("自分 0:05-0:15")

    def test_rejects_missing_range(self):
        with pytest.raises(ValueError):
            parse_enroll_spec("自分=0:05")

    def test_rejects_reversed_range(self):
        with pytest.raises(ValueError):
            parse_enroll_spec("自分=0:15-0:05")

    def test_rejects_zero_length_range(self):
        with pytest.raises(ValueError):
            parse_enroll_spec("自分=0:05-0:05")


class TestRelabelDissolved:
    def test_dissolved_speakers_lose_their_name(self):
        rows = [
            {"speaker": "不明話者7", "confident": False, "text": "a"},
            {"speaker": "不明話者6", "confident": False, "text": "b"},
            {"speaker": "自分", "confident": True, "text": "c"},
        ]
        count = replay_meeting.relabel_dissolved(rows, ["不明話者7"])
        assert count == 1
        assert [r["speaker"] for r in rows] == ["不明話者?", "不明話者6", "自分"]

    def test_nothing_to_relabel(self):
        rows = [{"speaker": "自分", "confident": True, "text": "c"}]
        assert replay_meeting.relabel_dissolved(rows, []) == 0


class TestReplayRenameAliases:
    """改名後に旧ラベルで届く発話が参加者統計に旧名を復活させない（aa の対照実験 2026-09-09）。"""

    def _actions(self, tmp_path):
        from types import SimpleNamespace

        diarizer = SimpleNamespace(
            enrolled_names=["自分"], unknown_names=[], speaker_names=["自分"],
            rename=lambda old, new: None,
        )
        return replay_meeting.ReplayUiActions(diarizer, tmp_path, {})

    def _seg(self, speaker, start):
        from src.stt.whisper_client import TranscriptSegment

        return TranscriptSegment(speaker, "テキスト", start, start + 1.0, "")

    def test_old_label_after_rename_is_folded_into_new_name(self, tmp_path):
        actions = self._actions(tmp_path)
        actions.record_segment(self._seg("自分", 0.0))
        actions.record_segment(self._seg("自分", 1.0))
        actions.rename_speaker("自分", "MYSELF")
        actions.record_segment(self._seg("自分", 2.0))   # 改名の瞬間に処理中だった発話
        assert {n: v[0] for n, v in actions.stats.items()} == {"MYSELF": 3}

    def test_rename_back_and_forth_then_late_old_label(self, tmp_path):
        actions = self._actions(tmp_path)
        actions.record_segment(self._seg("自分", 0.0))
        actions.rename_speaker("自分", "MYSELF")
        actions.rename_speaker("MYSELF", "自分")
        actions.record_segment(self._seg("MYSELF", 2.0))
        assert {n: v[0] for n, v in actions.stats.items()} == {"自分": 2}

    def test_ghost_participant_rename_updates_alias_and_stats_without_diarizer(self, tmp_path):
        """diarizer に無い行単位の話者も安全に改名できる。"""
        actions = self._actions(tmp_path)
        actions.stats["とりのこ"] = [1, 4]
        actions.rename_speaker("とりのこ", "鳥の子")
        assert actions._aliases["とりのこ"] == "鳥の子"
        assert actions.stats == {"鳥の子": [1, 4]}

    def test_relabel_label_is_preserved_and_renamed(self, tmp_path):
        """一行の付け替えラベルは改名時に追従する。"""
        actions = self._actions(tmp_path)
        actions.relabel_segment(1.001, 2.0, "不明話者?", "とりのこ")
        assert actions._row_labels == {1.0: "とりのこ"}
        actions.rename_speaker("とりのこ", "鳥の子")
        assert actions._row_labels == {1.0: "鳥の子"}

    def test_participants_hides_empty_ghost_participant(self, tmp_path):
        """発話を全て移した diarizer 外の名前は一覧から消える。"""
        actions = self._actions(tmp_path)
        actions.stats["とりのこ"] = [0, 0]
        assert "とりのこ" not in {item["name"] for item in actions.participants()}
