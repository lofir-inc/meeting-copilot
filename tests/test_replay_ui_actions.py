"""3f: 本物の ReplayUiActions と EnrolledDiarizer を通す検査。

phase3.py は FakeActions しか通しておらず、運用者 が踏んだ (a)〜(e) を素通りさせた（aa の対照実験 2026-09-09:
3f 前の実装に差し替えると 7/10 が FAIL する）。resemblyzer も whisper も LLM も呼ばず、Speaker を直に注入する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.replay_meeting import ReplayUiActions
from src.audio.enrolled_diarizer import DiarizerConfig, EnrolledDiarizer, Speaker
from src.stt.whisper_client import TranscriptSegment


def make_actions(tmp: Path) -> ReplayUiActions:
    """自分（登録）・不明話者1（検出）を持つ diarizer で actions を作る。"""
    diarizer = EnrolledDiarizer(DiarizerConfig())
    rng = np.random.default_rng(0)
    for name, enrolled in (("自分", True), ("不明話者1", False)):
        diarizer._speakers[name] = Speaker(name=name, enrolled=enrolled, enroll_embeddings=[rng.normal(size=256).astype(np.float32)])
    return ReplayUiActions(diarizer, tmp)


def seg(speaker: str, start: float, text: str = "テスト発話") -> TranscriptSegment:
    return TranscriptSegment(speaker=speaker, text=text, start_time=start, end_time=start + 2.0, timestamp=start)


def names(actions: ReplayUiActions) -> list[str]:
    return [item["name"] for item in actions.participants()]


def test_relabel_is_kept_on_server_with_rounded_key(tmp_path: Path) -> None:
    """(b)(②) 行の付け替えはサーバーに残り、キーは round(start_time, 2)。"""
    actions = make_actions(tmp_path)
    actions.record_segment(seg("不明話者?", 10.0))
    actions.relabel_segment(10.0, 12.0, "不明話者?", "とりのこ")
    assert actions._row_labels == {10.0: "とりのこ"}


def test_ghost_participant_can_be_renamed(tmp_path: Path) -> None:
    """(d) 付け替えで生まれた幽霊参加者（diarizer に無い）を改名しても例外にならない。"""
    actions = make_actions(tmp_path)
    actions.record_segment(seg("不明話者?", 20.0))
    actions.relabel_segment(20.0, 22.0, "不明話者?", "とりのこ")
    actions.rename_speaker("とりのこ", "鳥の子")
    assert "鳥の子" in names(actions) and "とりのこ" not in names(actions)
    assert actions._row_labels[20.0] == "鳥の子"


def test_known_speaker_rename_still_reaches_diarizer(tmp_path: Path) -> None:
    """(d') 登録・検出話者の改名は従来どおり diarizer に届く。"""
    actions = make_actions(tmp_path)
    actions.rename_speaker("不明話者1", "小島")
    assert "小島" in actions.diarizer.speaker_names and "不明話者1" not in actions.diarizer.speaker_names


def test_relabel_back_to_unknown_removes_ghost(tmp_path: Path) -> None:
    """(e) 不明話者? へ戻すと幽霊参加者は一覧から消える。"""
    actions = make_actions(tmp_path)
    actions.record_segment(seg("不明話者?", 30.0))
    actions.relabel_segment(30.0, 32.0, "不明話者?", "とりのこ")
    assert "とりのこ" in names(actions)
    actions.relabel_segment(30.0, 32.0, "とりのこ", "不明話者?")
    assert "とりのこ" not in names(actions)


@pytest.mark.parametrize("rename_first", [False, True])
def test_row_correction_beats_global_rename(tmp_path: Path, rename_first: bool) -> None:
    """(①)(①') 行の訂正は全体改名の前後どちらでも勝つ。"""
    actions = make_actions(tmp_path)
    actions.record_segment(seg("不明話者?", 40.0))
    if rename_first:
        actions.rename_speaker("不明話者?", "X")
        actions.relabel_segment(40.0, 42.0, "X", "とりのこ")
    else:
        actions.relabel_segment(40.0, 42.0, "不明話者?", "とりのこ")
        actions.rename_speaker("不明話者?", "X")
    assert actions._row_labels[40.0] == "とりのこ"


@pytest.mark.parametrize("start", [80.0, 60.007])
def test_redelivered_segment_carries_row_correction(tmp_path: Path, start: float) -> None:
    """(b')(②) 訂正後に同じ発話が再配信されると訂正名で届く（小数 3 桁でも）。"""
    actions = make_actions(tmp_path)
    actions.record_segment(seg("不明話者?", start))
    actions.relabel_segment(start, start + 2.0, "不明話者?", "とりのこ")
    again = seg("不明話者?", start)
    actions.record_segment(again)
    assert again.speaker == "とりのこ"


def test_relabel_preserves_total_utterances(tmp_path: Path) -> None:
    """付け替えで発話数の総和は変わらない。"""
    actions = make_actions(tmp_path)
    for index in range(3):
        actions.record_segment(seg("不明話者?", 70.0 + index))
    before = sum(item["utterances"] for item in actions.participants())
    actions.relabel_segment(70.0, 72.0, "不明話者?", "とりのこ")
    assert sum(item["utterances"] for item in actions.participants()) == before
