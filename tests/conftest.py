"""テスト全体の前提。"""

from __future__ import annotations

import os
import stat
import tempfile
import subprocess
from pathlib import Path

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


@pytest.fixture(autouse=True)
def _no_real_gcloud(monkeypatch):
    """gcloud は各テストの偽物だけを使い、認証や外部への問い合わせを実行しない。"""
    real_popen = subprocess.Popen

    def guarded_popen(command, *args, **kwargs):
        executable = command[0] if isinstance(command, (list, tuple)) else str(command).split()[0]
        if Path(executable).name == "gcloud":
            raise AssertionError("テストでは gcloud を偽物にしてください")
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)


@pytest.fixture(autouse=True)
def _restore_umask():
    """CLI entry points change process state; never leak it into another test."""
    previous = os.umask(0o077)
    os.umask(previous)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture(autouse=True)
def _no_real_directory_chmod(monkeypatch, tmp_path_factory, tmp_path):
    """Only pytest's temporary directories may have their permissions changed."""
    temporary_root = tmp_path_factory.getbasetemp().resolve()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    real_chmod, real_fchmod = os.chmod, os.fchmod

    def guarded_chmod(path, mode, *args, **kwargs):
        if isinstance(path, int):
            return guarded_fchmod(path, mode)
        target = Path(path).resolve()
        if target.is_dir() and not target.is_relative_to(temporary_root):
            raise AssertionError(f"テスト用フォルダの外の chmod を禁止しています: {target}")
        return real_chmod(path, mode, *args, **kwargs)

    def guarded_fchmod(descriptor, mode):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise AssertionError("テストではフォルダの fchmod を使わないでください")
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "chmod", guarded_chmod)
    monkeypatch.setattr(os, "fchmod", guarded_fchmod)


@pytest.fixture(autouse=True)
def _isolated_ui_token(monkeypatch, tmp_path):
    """★画面の認証トークン（~/.config/meeting-copilot/ui-token）を、テストが作ったり読んだりしないようにする。"""
    from src.ui import guard

    monkeypatch.setattr(guard, "TOKEN_PATH", tmp_path / "ui-config" / "ui-token")


def pytest_collection_modifyitems(config, items):
    """Native PDFKit requires WindowServer access unavailable in the sandbox."""
    if not os.environ.get("CODEX_SANDBOX"):
        return
    for item in items:
        if ((item.nodeid.startswith("tests/test_documents.py") and "pdf" in item.name.lower())
                or item.nodeid == "tests/test_meeting_orchestrator.py::TestPrepFromTheDashboard::test_文字の入っていない資料は断る"):
            item.add_marker(pytest.mark.skip(reason="PDFKit requires WindowServer outside the sandbox"))

        if item.nodeid.startswith("tests/test_ui_server.py::TestPortFallback::"):
            item.add_marker(pytest.mark.skip(reason="Listening sockets are unavailable in the sandbox"))
        if item.nodeid == "tests/test_meeting_orchestrator.py::TestLiveSttLabel::test_手元へ落ちたら表示も手元に戻る":
            item.add_marker(pytest.mark.skip(reason="MLX needs Metal outside the sandbox"))
