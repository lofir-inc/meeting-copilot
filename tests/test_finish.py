"""会議のあとの仕上げは、押してから走らせる（あとから再開できる）。

2026-09-16 運用者 指示: 対面の商談で終わったらすぐ移動する・Wi-Fi が切れる場面がある。
数分かかる処理が勝手に始まって途中で落ちるのを避ける。
"""

from __future__ import annotations

import json

from src import finish


def _session(tmp_path, **files):
    for name, body in files.items():
        (tmp_path / name.replace("__", ".")).write_text(body, encoding="utf-8")
    return tmp_path


def test_終わっている工程は残りに出ない(tmp_path):
    _session(tmp_path, transcripts_final__jsonl="{}", glossary_candidates__json="[]")

    assert [step.key for step in finish.remaining(tmp_path)] == ["voices", "minutes"]


def test_作り直しを使わない設定なら聞かない(tmp_path):
    assert [step.key for step in finish.remaining(tmp_path, wanted=["voices", "glossary"])] == ["voices", "glossary"]


def test_議事録は貼り付け用の指示書でも終わり扱い(tmp_path):
    """手段が無くて指示書になった会議を、いつまでも「残り」に出さない。"""
    _session(tmp_path, minutes_prompt__md="# 指示書")

    assert "minutes" not in [step.key for step in finish.remaining(tmp_path)]


def test_残りを書いて読んで消す(tmp_path):
    finish.write_pending(tmp_path, finish.remaining(tmp_path), note="あとでやるを選んだ")

    pending = finish.read_pending(tmp_path)
    assert pending["note"] == "あとでやるを選んだ"
    assert [step["key"] for step in pending["steps"]] == ["finalize", "voices", "glossary", "minutes"]
    assert json.loads((tmp_path / finish.PENDING_FILE).read_text(encoding="utf-8"))["steps"]

    finish.clear_pending(tmp_path)
    assert finish.read_pending(tmp_path) is None


def test_ここまでは残っていると言える中身(tmp_path):
    _session(tmp_path, transcripts__jsonl="{}", minutes_input__md="# 材料")
    (tmp_path / "recording_remote.wav").write_bytes(b"RIFF")
    (tmp_path / "state.json").write_text("", encoding="utf-8")   # 空は数えない

    assert finish.saved_already(tmp_path) == ["相手側の録音", "会議中の全文", "議事録の材料"]
