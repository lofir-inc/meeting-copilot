"""エントリポイントの小さな関数のテスト。"""

import os
from datetime import date, datetime

from pathlib import Path

from src.main import setup_session, stale_prep_files, stale_prep_warnings


def _touch(path, day: date) -> None:
    path.write_text("# 資料\n", encoding="utf-8")
    stamp = datetime(day.year, day.month, day.day, 12, 0).timestamp()
    os.utime(path, (stamp, stamp))


def test_old_prep_is_reported_with_its_age(tmp_path):
    """2026-09-11 本番: prep に 4 月の資料が残っていて、今日の会議の材料になっていた。"""
    _touch(tmp_path / "2026-04-11-toria.md", date(2026, 4, 11))
    _touch(tmp_path / "today.md", date(2026, 9, 11))
    assert stale_prep_files(tmp_path, date(2026, 9, 11)) == [("2026-04-11-toria.md", 153)]


def test_missing_prep_dir_is_fine(tmp_path):
    assert stale_prep_files(tmp_path / "none", date(2026, 9, 11)) == []


def test_古い資料の警告は画面にも出せる形で返る(tmp_path):
    """ターミナルだけだと気づけない（運用者 は画面を見て会議をする）。"""
    warnings = stale_prep_warnings([("2026-04-11-toria.md", 153)], tmp_path)
    assert len(warnings) == 1
    assert "2026-04-11-toria.md（153 日前）" in warnings[0]
    assert "要約と問いはこの資料を基準にします" in warnings[0]


def test_今日の資料だけなら警告は出ない(tmp_path):
    assert stale_prep_warnings([], tmp_path) == []


def test_会議ごとにリセットするとコピーしない(tmp_path):
    """共有フォルダの資料を使い回さない（2026-09-11 は 4 月の計画が基準になっていた）。"""
    shared = tmp_path / "shared"; shared.mkdir()
    (shared / "2026-04-11-toria.md").write_text("# 4 月の計画\n", encoding="utf-8")
    session = tmp_path / "session"

    setup_session(session, None)

    assert (session / "prep").is_dir()          # 置き場は作るが
    assert list((session / "prep").iterdir()) == []   # 中身は空から始まる


def test_明示して渡せば従来どおりコピーする(tmp_path):
    shared = tmp_path / "shared"; shared.mkdir()
    (shared / "議題.md").write_text("# 議題\n", encoding="utf-8")
    session = tmp_path / "session"

    setup_session(session, shared)

    assert [path.name for path in (session / "prep").iterdir()] == ["議題.md"]
