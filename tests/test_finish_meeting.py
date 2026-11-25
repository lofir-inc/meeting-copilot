import json
import subprocess

import pytest

from scripts import finish_meeting


@pytest.mark.parametrize("lines,approved", [
    ([], False),
    ([{"note": "live_batch", "billing_project": "fake-project"}], False),
    ([{"note": "live_batch", "purposes": ["live_batch"], "billing_project": "fake-project"}], False),
    ([{"note": "state", "purposes": ["state", "prep"], "billing_project": "fake-project"}], False),
    ([{"note": "verify", "purposes": ["verify"], "billing_project": "fake-project"},
      {"note": "ask", "purposes": ["ask"], "billing_project": "fake-project"}], False),
    ([{"note": "state", "purposes": ["state", "prep", "minutes"], "billing_project": "fake-project"}], True),
    ([{"purposes": ["minutes"], "billing_project": "deepgram"}], False),
    ([{"purposes": ["minutes"], "billing_project": "gemini"}], False),
    ([{"purposes": ["minutes"], "billing_project": ""}], False),
    ([{"purposes": ["minutes"], "billing_project": None}], False),
    ([{"purposes": ["minutes"], "billing_project": "not a project"}], False),
    ([{"purposes": "minutes", "billing_project": "fake-project"}], False),
    ([{"note": "state", "billing_project": "fake-project"}], False),
    (["broken", [], None, {"purposes": ["minutes"], "billing_project": 123}], False),
    (["broken", {"purposes": ["minutes"], "billing_project": "fake-project"}], True),
])
def test_minutes_send_approval(tmp_path, monkeypatch, lines, approved):
    settings = {"meeting": {"external_stt": {"billing_project": "fake-project"}}}
    if lines:
        (tmp_path / "external_sends.jsonl").write_text(
            "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines))
    calls = []
    monkeypatch.setattr(finish_meeting.subprocess, "run", lambda argv, **kwargs: (
        calls.append(argv) or subprocess.CompletedProcess(argv, 0)))
    assert finish_meeting.minutes_send_approved(tmp_path, settings) is approved
    assert finish_meeting.make_minutes(tmp_path, settings)
    assert ("--approved-external" if approved else "--no-external") in calls[0]


@pytest.mark.parametrize("settings,approved", [
    ({}, False),
    ({"meeting": None}, False),
    ({"meeting": {}}, False),
    ({"meeting": {"external_stt": None}}, False),
    ({"meeting": {"external_stt": {}}}, False),
    ({"meeting": {"external_stt": {"billing_project": ""}}}, False),
    ({"meeting": {"external_stt": {"billing_project": None}}}, False),
    ({"meeting": {"external_stt": {"billing_project": "other-project"}}}, False),
    ({"meeting": {"external_stt": {"billing_project": "fake-project"}}}, True),
])
def test_minutes_approval_matches_current_destination(tmp_path, monkeypatch, settings, approved):
    (tmp_path / "external_sends.jsonl").write_text(json.dumps(
        {"purposes": ["minutes"], "billing_project": "fake-project"}))
    calls = []
    monkeypatch.setattr(finish_meeting.subprocess, "run", lambda argv, **kwargs: (
        calls.append(argv) or subprocess.CompletedProcess(argv, 0)))

    assert finish_meeting.minutes_send_approved(tmp_path, settings) is approved
    assert finish_meeting.make_minutes(tmp_path, settings)
    assert ("--approved-external" if approved else "--no-external") in calls[0]


def test_minutes_approval_can_find_matching_destination_after_old_record(tmp_path):
    (tmp_path / "external_sends.jsonl").write_text("\n".join(json.dumps(
        {"purposes": ["minutes"], "billing_project": project})
        for project in ["old-project", "current-project"]))
    settings = {"meeting": {"external_stt": {"billing_project": "current-project"}}}

    assert finish_meeting.minutes_send_approved(tmp_path, settings)
