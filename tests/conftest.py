"""テスト全体の前提。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_finder_windows(monkeypatch):
    """テストで Finder のウィンドウを開かない（会議の終了処理が成果物フォルダを開くため）。

    2026-09-16: テストを流すたびに pytest の一時フォルダが Finder で開いていた（運用者 指摘）。
    """
    monkeypatch.setenv("MEETING_NO_OPEN", "1")


@pytest.fixture(autouse=True)
def _no_real_billing_credential(monkeypatch, tmp_path):
    """この Mac に置いた課金確認用の鍵を、テストが読みにいかないようにする（本物の gcloud を呼ばない）。"""
    monkeypatch.setenv("MEETING_BILLING_CREDENTIAL", str(tmp_path / "no-billing-credential.json"))
