"""Sensitive files stay private, even when the caller's umask is permissive."""

import json
import os
import sqlite3
import stat
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import macwhisper_reference
from src import private_files
from src.app import meeting_launch
from src.audio.enrolled_diarizer import EnrolledDiarizer
from src.audio.voice_library import VoiceLibrary
from src.main import setup_session


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def permissive_umask():
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


def test_private_umask_protects_new_files_and_directories(tmp_path, permissive_umask):
    private_files.private_umask()
    path = tmp_path / "transcripts.jsonl"
    path.write_text("private")
    directory = tmp_path / "audio"
    directory.mkdir()
    assert mode(path) == 0o600
    assert mode(directory) == 0o700


def test_setup_session_protects_parent_without_touching_old_contents(tmp_path, permissive_umask):
    sessions = tmp_path / "workspace" / "sessions"
    old = sessions / "old"
    old.mkdir(parents=True)
    old.chmod(0o755)
    transcript = old / "transcripts.jsonl"
    transcript.write_text("old transcript")
    transcript.chmod(0o644)
    sessions.chmod(0o755)

    new = sessions / "new"
    setup_session(new, None)

    assert mode(sessions) == mode(new) == 0o700
    assert mode(old) == 0o755
    assert mode(transcript) == 0o644
    assert transcript.read_text() == "old transcript"
    # Resuming an existing meeting does not recursively change old permissions.
    setup_session(old, None)
    assert mode(old) == 0o755
    assert mode(transcript) == 0o644


def test_custom_session_does_not_chmod_arbitrary_parent(tmp_path, permissive_umask):
    parent = tmp_path / "custom"
    parent.mkdir()
    parent.chmod(0o755)
    setup_session(parent / "meeting", None)
    assert mode(parent) == 0o755
    assert mode(parent / "meeting") == 0o700


@pytest.mark.parametrize("existing", [False, True])
def test_speakers_json_is_private(tmp_path, permissive_umask, existing):
    path = tmp_path / "speakers.json"
    if existing:
        path.write_text("old")
        path.chmod(0o644)
    EnrolledDiarizer().save(path)
    assert mode(path) == 0o600
    assert json.loads(path.read_text())["speakers"] == []


@pytest.mark.parametrize("existing_trash", [False, True])
def test_forgotten_voice_and_trash_are_private(tmp_path, permissive_umask, existing_trash):
    library = VoiceLibrary(tmp_path / "library.json")
    library.people["Alice"] = [{"session": "meeting", "embeddings": [[1.0, 0.0]]}]
    trash = tmp_path / "trash"
    if existing_trash:
        trash.mkdir()
        trash.chmod(0o755)
    target = library.forget("Alice", trash)
    assert mode(trash) == 0o700
    assert mode(target) == 0o600
    assert json.loads(target.read_text())["name"] == "Alice"
    assert "Alice" not in library.people




@pytest.fixture
def macwhisper_db(tmp_path, monkeypatch):
    source = tmp_path / "database"
    source.mkdir()
    with sqlite3.connect(source / "main.sqlite") as db:
        db.execute("create table transcript (text)")
        db.execute("insert into transcript values ('secret')")
    (source / "main.sqlite").chmod(0o644)
    (source / "main.sqlite-wal").write_bytes(b"sidecar")
    temporary = tmp_path / "copies"
    temporary.mkdir()
    monkeypatch.setattr(macwhisper_reference, "DB_DIR", source)
    monkeypatch.setattr(macwhisper_reference.tempfile, "tempdir", str(temporary))
    return source, temporary


@pytest.mark.parametrize("fail", [False, True])
def test_db_copy_lives_until_read_then_is_removed(macwhisper_db, permissive_umask, fail):
    source, temporary = macwhisper_db
    with pytest.raises(RuntimeError) if fail else nullcontext():
        with macwhisper_reference._copy_db() as copied:
            folder = copied.parent
            assert mode(folder) == 0o700
            assert mode(copied) == 0o600
            assert mode(folder / "main.sqlite-wal") == 0o600
            assert copied.read_bytes() == (source / "main.sqlite").read_bytes()
            if fail:
                raise RuntimeError("reader failed")
    assert not folder.exists()
    assert list(temporary.iterdir()) == []
    assert (source / "main.sqlite").exists()


def test_missing_db_leaves_no_copy(macwhisper_db):
    source, temporary = macwhisper_db
    (source / "main.sqlite").unlink()
    with pytest.raises(SystemExit):
        with macwhisper_reference._copy_db():
            pytest.fail("missing database was accepted")
    assert list(temporary.iterdir()) == []


def test_failed_db_copy_is_cleaned_up(macwhisper_db, monkeypatch):
    _, temporary = macwhisper_db

    def fail(*args):
        raise OSError("copy failed")

    monkeypatch.setattr(macwhisper_reference.shutil, "copyfileobj", fail)
    with pytest.raises(OSError):
        with macwhisper_reference._copy_db():
            pytest.fail("failed copy was accepted")
    assert list(temporary.iterdir()) == []


def test_voice_library_temporary_file_is_private_before_replace(tmp_path, permissive_umask, monkeypatch):
    from pathlib import Path

    library = VoiceLibrary(tmp_path / "library.json")
    temporary = library.path.with_suffix(".tmp")
    temporary.write_text("old")
    temporary.chmod(0o644)
    real_replace = Path.replace
    observed = []

    def replace(path, target):
        observed.append(mode(path))
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    library.save()
    assert observed == [0o600]
    assert mode(library.path) == 0o600
    assert not temporary.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_meeting_child_inherits_umask_and_log_is_private(tmp_path, permissive_umask, existing):
    if existing:
        log = meeting_launch.log_path(tmp_path)
        log.parent.mkdir()
        log.write_text("old log")
        log.chmod(0o644)
    private_files.private_umask()
    child_file = tmp_path / "child-transcript.jsonl"

    def spawn(command, **kwargs):
        # Exercise the real process creation options, replacing only the meeting.
        return subprocess.Popen([sys.executable, "-c",
                                 "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('private'); print('started')",
                                 str(child_file)], **kwargs)

    process = meeting_launch.start(tmp_path, sys.executable, popen=spawn)
    assert process.wait(timeout=10) == 0
    assert mode(child_file) == 0o600
    assert mode(meeting_launch.log_path(tmp_path)) == 0o600
    assert meeting_launch.log_path(tmp_path).read_text().strip() == "started"


def test_bench_asr_main_creates_private_results(tmp_path, permissive_umask, monkeypatch):
    from scripts import bench_asr

    reference = tmp_path / "reference.txt"
    reference.write_text("private transcript")
    output = tmp_path / "results.json"
    rows = [{"engine": "local", "text": "private transcript", "text_raw": "private transcript",
             "error_rate": 0, "seconds": 1, "speed": 1, "yen_per_hour": 0, "error": ""}]
    monkeypatch.setattr(bench_asr, "bench", lambda *args, **kwargs: rows)
    monkeypatch.setattr(bench_asr, "audio_seconds", lambda path: 1)
    monkeypatch.setattr(sys, "argv", ["bench_asr", "--audio", str(tmp_path / "audio.wav"),
                                     "--reference", str(reference), "--out", str(output)])

    assert bench_asr.main() == 0
    assert mode(output) == 0o600
    assert json.loads(output.read_text()) == rows


def test_sck_probe_main_creates_private_recording(tmp_path, permissive_umask, monkeypatch):
    import numpy as np
    from scripts import sck_probe

    class Capture:
        def __init__(self, config, callback):
            self.cfg = SimpleNamespace(sample_rate=1000)
            self._reblocker = SimpleNamespace(flush=lambda: None)
            self.first_frame_at = None
            self.callback = callback

        def start(self):
            self.callback(np.ones(100, dtype=np.float32) * 0.1)

        def stop(self):
            pass

        def stats(self):
            return dict(buffer_pts=[], buffer_samples=[], queue_mode="test", buffers=1,
                        gap_max_sec=0, gap_total_sec=0, pts_vs_samples_drift_sec=0, format_flags=0)

    monkeypatch.setitem(sys.modules, "src.audio.sck_capture", SimpleNamespace(
        SckConfig=lambda **kwargs: None, SckSystemAudio=Capture,
        has_screen_capture_permission=lambda: True, responsible_app_name=lambda: "test"))
    monkeypatch.setattr(sck_probe, "REPO", tmp_path)
    monkeypatch.setattr(sck_probe, "SckConfig", lambda **kwargs: None)
    monkeypatch.setattr(sck_probe, "SckSystemAudio", Capture)
    monkeypatch.setattr(sck_probe, "has_screen_capture_permission", lambda: True)
    monkeypatch.setattr(sck_probe, "responsible_app_name", lambda: "test")
    output = tmp_path / "recording" / "probe.wav"
    monkeypatch.setattr(sys, "argv", ["sck_probe", "--seconds", "0", "--out", str(output), "--json"])

    assert sck_probe.main() == 0
    assert mode(output) == 0o600
    assert mode(output.parent) == 0o700
    assert output.stat().st_size > 44
    assert mode(tmp_path / "workspace" / "accept" / "sck_probe.json") == 0o600


@pytest.mark.parametrize("existing_work", [False, True])
def test_rebuild_external_copies_contents_privately(tmp_path, permissive_umask, monkeypatch, existing_work):
    from scripts import rebuild_external

    sessions = tmp_path / "sessions"
    source = sessions / "original"
    source.mkdir(parents=True)
    sessions.chmod(0o755)
    source.chmod(0o755)
    contents = {"speakers.json": b'{"speakers": []}', "corrections.jsonl": b'{"text": "private"}\n'}
    for name, content in contents.items():
        origin = source / name
        origin.write_bytes(content)
        origin.chmod(0o644)
        os.utime(origin, ns=(1_700_000_000_000_000_000, 1_700_000_001_000_000_000))
    recording = source / "recording_remote.wav"
    recording.write_bytes(b"recording")
    work = sessions / "original-gemini"
    if existing_work:
        work.mkdir()
        work.chmod(0o755)
    monkeypatch.setattr(rebuild_external, "SESSIONS", sessions)

    assert rebuild_external.prepare(source) == work
    assert mode(sessions) == 0o700
    assert mode(work) == (0o755 if existing_work else 0o700)
    assert mode(source) == 0o755
    for name, content in contents.items():
        assert (work / name).read_bytes() == content
        assert mode(work / name) == 0o600
        assert mode(source / name) == 0o644
        assert (work / name).stat().st_mtime_ns == (source / name).stat().st_mtime_ns
    assert (work / recording.name).resolve() == recording


@pytest.mark.parametrize("stale", [False, True])
def test_state_atomic_write_is_private(tmp_path, permissive_umask, stale):
    from src.llm.meeting_state import MeetingState
    from src.output.markdown_writer import MarkdownWriter, OutputConfig
    temporary = tmp_path / "state.json.tmp"
    if stale:
        temporary.write_text("old")
        temporary.chmod(0o644)
    MarkdownWriter(OutputConfig(workspace_dir=str(tmp_path))).write_state_json(MeetingState(summary="private"))
    assert mode(tmp_path / "state.json") == 0o600
    assert not temporary.exists()


@pytest.mark.parametrize("operation", ["meetings", "wav"])
@pytest.mark.parametrize("existing_trash", [False, True])
def test_trash_is_private_before_moving_old_data(tmp_path, permissive_umask, monkeypatch, operation, existing_trash):
    from datetime import datetime
    import shutil
    from src.audio import archive
    from src.ui import library

    session = tmp_path / "workspace" / "sessions" / "old"
    session.mkdir(parents=True)
    session.parent.chmod(0o700)
    session.chmod(0o755)
    recording = session / "recording_remote.wav"
    recording.write_bytes(b"old recording")
    recording.chmod(0o644)
    trash_root = tmp_path / "workspace" / "99-trash"
    clock = SimpleNamespace(now=lambda tz: datetime(2026, 11, 25, 12, tzinfo=tz))
    if operation == "meetings":
        lib = library.Library(tmp_path, lambda: {})
        monkeypatch.setattr(library, "datetime", clock)
        destination = trash_root / "2026-11-25_sessions"
        move = lambda: lib.delete_meetings([session.name])
        moved_recording = destination / session.name / recording.name
    else:
        (session / "audio").mkdir()
        (session / "audio" / "remote.mp3").write_bytes(b"mp3")
        monkeypatch.setattr(archive, "datetime", clock)
        destination = trash_root / "2026-11-25_old-wav"
        move = lambda: archive.drop_wav(session, trash_root)
        moved_recording = destination / recording.name
    if existing_trash:
        destination.mkdir(parents=True)
        destination.chmod(0o755)
    real_move = shutil.move
    observed = []

    def checked_move(source, target, *args, **kwargs):
        observed.append(mode(destination))
        return real_move(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "move", checked_move)
    move()

    assert observed == [0o700]
    assert mode(destination) == 0o700
    assert moved_recording.read_bytes() == b"old recording"
    assert mode(moved_recording) == 0o644
    assert mode(moved_recording.parent) == (0o755 if operation == "meetings" else 0o700)
    assert not recording.exists()




def test_screen_capture_reuses_stale_temporary_privately(tmp_path, permissive_umask, monkeypatch):
    from src.screen import capture

    temporary = tmp_path / "screens" / "_tmp.jpg"
    temporary.parent.mkdir()
    temporary.write_bytes(b"stale")
    temporary.chmod(0o644)
    monkeypatch.setattr(capture, "find_window", lambda config: {"id": 1, "title": "Zoom", "owner": "zoom.us"})
    monkeypatch.setattr(capture, "fingerprint", lambda path: b"new image")

    def take_picture(window_id, path, **kwargs):
        assert mode(path) == 0o600
        path.write_bytes(b"new image")
        return True

    monkeypatch.setattr(capture, "capture_window", take_picture)
    watcher = capture.ScreenWatcher(tmp_path, capture.ScreenConfig(enabled=True, ocr=False), None)
    watcher._meeting_clock = lambda: 12
    result = watcher.tick()
    target = tmp_path / result["file"]
    assert mode(target) == 0o600
    assert target.read_bytes() == b"new image"
    assert not temporary.exists()
