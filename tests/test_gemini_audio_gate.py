"""音声の実際の送り先が Gemini なら、provider に依存せず Google を確認する。"""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import yaml

from scripts import (
    bench_asr, finalize_meeting, gemini_live_transcribe, gemini_transcribe, probe_live, selftest_live,
)
from src import meeting_orchestrator as mo
from src.stt import external_consent as consent
from src.stt import gemini_transcribe as transcribe
from src.stt import deepgram_transcribe as deepgram
from tests.test_meeting_orchestrator import orch  # noqa: F401 — fixture


@pytest.fixture
def external(tmp_path, monkeypatch):
    key = tmp_path / "gemini.key"
    key.write_text("verified-audio-key\n", encoding="utf-8")
    dg_key = tmp_path / "deepgram.key"
    dg_key.write_text("fake-deepgram-key", encoding="utf-8")
    cfg = consent.ExternalSttConfig(
        enabled=True, provider="deepgram", billing_project="fake-project",
        key_file=str(key), deepgram_key_file=str(dg_key),
    )
    account = Mock(return_value=SimpleNamespace(data_use=lambda: {"ok": True, "why": "paid"}))
    monkeypatch.setattr(deepgram, "DeepgramApi", account)
    return cfg


def prepare_entry(entry, cfg, tmp_path, monkeypatch, orch):
    """各入口を実際に通し、音声モデル・ネットワークだけを偽物にする。"""
    settings = tmp_path / "config" / "settings.yaml"
    settings.parent.mkdir()
    settings.write_text(yaml.safe_dump({"meeting": {"external_stt": {
        "enabled": cfg.enabled, "provider": cfg.provider, "billing_project": cfg.billing_project,
        "key_file": cfg.key_file, "deepgram_key_file": cfg.deepgram_key_file,
    }}}), encoding="utf-8")
    session = tmp_path / "workspace" / "sessions" / "fake-audio"
    session.mkdir(parents=True)
    audio_path = session / "recording_remote.wav"
    audio_path.write_bytes(b"fake audio")
    audio = np.zeros(16000, dtype=np.float32)
    api = SimpleNamespace(
        transcribe_audio=Mock(return_value={"utterances": [], "usage": {}}),
        transcribe_file=Mock(return_value={"utterances": [], "usage": {}}),
    )
    made = Mock(return_value=api)

    if entry == "gaps":
        monkeypatch.setattr(finalize_meeting, "TranscribeApi", made)
        run = lambda: finalize_meeting.pick_up_gaps_external(
            session, [{"stream": "remote", "start_time": 0, "end_time": 1}], cfg,
            16000, {"remote": audio}, "自分", approved=True)
    elif entry == "live":
        cfg.provider = "gemini"
        orch._external_cfg = cfg
        orch._stt_engine = "gemini"
        monkeypatch.setattr(orch.whisper, "_ensure_model", lambda: None)
        monkeypatch.setattr(mo, "TranscribeApi", made)
        monkeypatch.setattr(mo, "LiveBatchStt", lambda *a, **kw: object())
        run = lambda: orch._start_live_stt(consent.Approval(billing_project=cfg.billing_project))
    elif entry == "selftest":
        monkeypatch.setattr(selftest_live, "REPO", tmp_path)
        monkeypatch.setattr(selftest_live, "decode_audio", lambda *a: audio)
        monkeypatch.setattr(selftest_live, "windows_of", lambda *a: [SimpleNamespace(
            audio=audio, sample_rate=16000, start_time=0, end_time=1, speech_sec=1, slice=lambda *a: audio)])
        monkeypatch.setattr(selftest_live, "TranscribeApi", made)
        monkeypatch.setattr(selftest_live, "EnrolledDiarizer", lambda *a: None)
        monkeypatch.setattr(selftest_live, "label_segments", lambda *a, **kw: None)
        monkeypatch.setattr(sys, "argv", ["selftest_live.py", "--config", str(settings),
                                         "--session", session.name, "--send", "--approved"])
        run = selftest_live.main
    elif entry == "bench":
        monkeypatch.setattr(bench_asr, "REPO", tmp_path)
        monkeypatch.setattr(transcribe, "TranscribeApi", made)
        run = lambda: bench_asr.run_gemini(tmp_path / "public.wav")
    elif entry == "socket":
        made = AsyncMock(return_value=[])
        monkeypatch.setattr(gemini_live_transcribe, "stream", made)
        monkeypatch.setattr(gemini_live_transcribe, "load_pcm16", Mock(return_value=b"fake-pcm"))
        monkeypatch.setattr(sys, "argv", ["gemini_live_transcribe.py", str(audio_path),
                                         "--out", str(session / "out.jsonl"), "--config", str(settings),
                                         "--key-file", cfg.key_file, "--billing-project", cfg.billing_project])
        run = gemini_live_transcribe.main
    elif entry == "probe":
        made = AsyncMock(return_value=0)
        monkeypatch.setattr(probe_live, "REPO", tmp_path)
        monkeypatch.setattr(probe_live, "probe", made)
        monkeypatch.setattr(sys, "argv", ["probe_live.py", "--audio", str(audio_path)])
        run = probe_live.main
    else:
        monkeypatch.setattr(gemini_transcribe, "TranscribeApi", made)
        monkeypatch.setattr(gemini_transcribe, "require_paid_project", lambda project: None)
        monkeypatch.setattr(gemini_transcribe, "audio_seconds", lambda path: 1)
        monkeypatch.setattr(gemini_transcribe, "to_mp3", lambda *a: None)
        monkeypatch.setattr(sys, "argv", ["gemini_transcribe.py", str(audio_path), "--out", str(session / "out.jsonl"),
                                         "--config", str(settings), "--key-file", cfg.key_file,
                                         "--billing-project", cfg.billing_project, "--keep-silence"])
        run = gemini_transcribe.main
    return run, made, api, session


@pytest.mark.parametrize("entry", ["gaps", "live", "selftest", "bench", "cli", "socket", "probe"])
@pytest.mark.parametrize("failure", ["unpaid", "error", "timeout"])
def test_all_audio_entries_refuse_before_constructing_api(external, tmp_path, monkeypatch, orch, entry, failure):
    run, made, api, session = prepare_entry(entry, external, tmp_path, monkeypatch, orch)
    billing = Mock(return_value=(False, "unpaid"), side_effect={
        "unpaid": None, "error": OSError("offline"),
        "timeout": subprocess.TimeoutExpired("fake-gcloud", 20),
    }[failure])
    monkeypatch.setattr(consent, "billing_enabled", billing)

    if entry in {"gaps", "bench"}:
        with pytest.raises(consent.ExternalSendRefused):
            run()
    else:
        assert run() == {"live": False, "selftest": 0, "cli": 1, "socket": 1, "probe": 1}[entry]
    made.assert_not_called()
    api.transcribe_audio.assert_not_called()
    api.transcribe_file.assert_not_called()
    billing.assert_called_once()
    assert billing.call_args.args == ("fake-project",)
    assert 0 < billing.call_args.kwargs["timeout_sec"] <= 20
    assert not (session / "external_sends.jsonl").exists()
    if entry == "gaps":
        assert external.provider == "deepgram"
        assert deepgram.DeepgramApi.call_count == 1  # Deepgram の承認を通っても Google で止まる
        log = (session / "interview.log").read_text(encoding="utf-8").splitlines()
        assert len(log) == 1 and "Gemini の音声送信を許可できません" in log[0]
    if entry == "live":
        assert "Gemini の音声送信を許可できません" in orch.bus.latest("status").data["warnings"][0]


@pytest.mark.parametrize("entry", ["gaps", "live", "selftest", "bench", "cli", "socket", "probe"])
def test_audio_entries_use_key_read_before_billing(external, tmp_path, monkeypatch, orch, entry):
    run, made, api, session = prepare_entry(entry, external, tmp_path, monkeypatch, orch)
    key_path = tmp_path / "gemini.key"

    def paid(*a, **kw):
        key_path.unlink()  # 確認後に読み直すと失敗する。環境変数へ逃げることも禁止。
        return True, "paid"

    monkeypatch.setenv("GEMINI_API_KEY", "unverified-env-key")
    monkeypatch.setattr(consent, "billing_enabled", paid)
    run()
    made.assert_called_once()
    assert made.call_args.args[1 if entry in {"socket", "probe"} else 0] == "verified-audio-key"
    if entry == "gaps":
        assert api.transcribe_audio.call_count == 1
        assert json.loads((session / "external_sends.jsonl").read_text())["note"] == "finalize_gaps"


@pytest.mark.parametrize("failure", ["project", "missing", "empty", "env"])
def test_gaps_reject_unknown_project_or_key(external, tmp_path, monkeypatch, orch, failure):
    if failure == "project":
        external.billing_project = ""
    elif failure == "missing":
        (tmp_path / "gemini.key").unlink()
    elif failure == "empty":
        (tmp_path / "gemini.key").write_text("", encoding="utf-8")
    else:
        external.key_file = ""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-env-key")
    run, made, _, _ = prepare_entry("gaps", external, tmp_path, monkeypatch, orch)
    billing = Mock(side_effect=AssertionError("不明なキーやプロジェクトでは問い合わせない"))
    monkeypatch.setattr(consent, "billing_enabled", billing)
    with pytest.raises(consent.ExternalSendRefused):
        run()
    made.assert_not_called()
    billing.assert_not_called()


def test_audio_key_provenance_and_text_decision_are_independent(external, tmp_path, monkeypatch):
    billing = Mock(return_value=(True, "paid"))
    monkeypatch.setattr(consent, "billing_enabled", billing)
    other = tmp_path / "other.key"
    other.write_text("unverified-key", encoding="utf-8")
    assert not consent.gemini_text_allowed({"key_file": str(other)}, external).allowed
    with pytest.raises(consent.ExternalSendRefused, match="キーの出どころ"):
        consent.require_gemini_audio_key(external, key_file=other)
    assert consent.require_gemini_audio_key(external) == "verified-audio-key"
    alias = tmp_path / "alias.key"
    alias.symlink_to(external.key_file)
    assert consent.require_gemini_audio_key(external, key_file=alias) == "verified-audio-key"
    billing.assert_called_once()


def test_text_and_audio_share_google_billing_result(external, monkeypatch):
    billing = Mock(return_value=(True, "paid"))
    monkeypatch.setattr(consent, "billing_enabled", billing)
    assert consent.gemini_text_allowed({"key_file": external.key_file}, external).allowed
    assert consent.require_gemini_audio_key(external) == "verified-audio-key"
    billing.assert_called_once()


@pytest.mark.parametrize("failure", ["different", "copied", "env", "project"])
@pytest.mark.parametrize("entry", ["cli", "socket"])
def test_cli_cannot_replace_configured_key_or_project(external, tmp_path, monkeypatch, orch, entry, failure):
    run, made, _, _ = prepare_entry(entry, external, tmp_path, monkeypatch, orch)
    argv = list(sys.argv)
    if failure in {"different", "copied"}:
        other = tmp_path / "other.key"
        other.write_text("verified-audio-key" if failure == "copied" else "unverified-key", encoding="utf-8")
        argv[argv.index("--key-file") + 1] = str(other)
    elif failure == "env":
        index = argv.index("--key-file")
        del argv[index:index + 2]
        monkeypatch.setenv("GEMINI_API_KEY", "unverified-key")
    else:
        argv[argv.index("--billing-project") + 1] = "other-project"
    monkeypatch.setattr(sys, "argv", argv)
    billing = Mock(side_effect=AssertionError("出どころが不明なら問い合わせない"))
    monkeypatch.setattr(consent, "billing_enabled", billing)
    if entry == "socket" and failure == "env":
        with pytest.raises(SystemExit) as exited:
            run()
        assert exited.value.code == 2  # --key-file は必須。環境変数へ逃がさない。
    else:
        assert run() == 1
    made.assert_not_called()
    billing.assert_not_called()


@pytest.mark.parametrize("provider", ["gemini", "deepgram"])
def test_live_rejects_arbitrary_key_even_when_cli_project_is_paid(external, tmp_path, monkeypatch, orch, provider):
    """指定した課金済みプロジェクトとキーの出どころを両方確認する。"""
    external.provider = provider
    run, made, _, _ = prepare_entry("socket", external, tmp_path, monkeypatch, orch)
    other = tmp_path / "free-project.key"
    other.write_text("fake-free-project-key", encoding="utf-8")
    argv = list(sys.argv)
    index = argv.index("--config")
    del argv[index:index + 2]  # 既存の CLI と同じ引数で、既定の設定も確認する。
    argv[argv.index("--key-file") + 1] = str(other)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(gemini_live_transcribe, "REPO", tmp_path)
    billing = Mock(return_value=(True, "paid"))
    monkeypatch.setattr(consent, "billing_enabled", billing)
    assert run() == 1
    made.assert_not_called()
    gemini_live_transcribe.load_pcm16.assert_not_called()
    billing.assert_not_called()


@pytest.mark.parametrize("entry", ["cli", "socket"])
def test_cli_accepts_symlink_to_configured_key(external, tmp_path, monkeypatch, orch, entry):
    run, made, _, _ = prepare_entry(entry, external, tmp_path, monkeypatch, orch)
    alias = tmp_path / "alias.key"
    alias.symlink_to(external.key_file)
    argv = list(sys.argv)
    argv[argv.index("--key-file") + 1] = str(alias)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(consent, "billing_enabled", lambda *a, **kw: (True, "paid"))
    assert run() == 0
    assert made.call_args.args[1 if entry == "socket" else 0] == "verified-audio-key"


@pytest.mark.parametrize("entry", ["socket", "probe"])
@pytest.mark.parametrize("failure", ["project", "missing", "empty", "env"])
def test_websocket_entries_refuse_unknown_project_or_key(external, tmp_path, monkeypatch, orch, entry, failure):
    if failure == "project":
        external.billing_project = ""
    elif failure == "missing":
        (tmp_path / "gemini.key").unlink()
    elif failure == "empty":
        (tmp_path / "gemini.key").write_text("", encoding="utf-8")
    else:
        external.key_file = ""
        monkeypatch.setenv("GEMINI_API_KEY", "unverified-env-key")
    run, made, _, _ = prepare_entry(entry, external, tmp_path, monkeypatch, orch)
    billing = Mock(side_effect=AssertionError("出どころが不明なら問い合わせない"))
    monkeypatch.setattr(consent, "billing_enabled", billing)
    assert run() == 1
    made.assert_not_called()
    billing.assert_not_called()
