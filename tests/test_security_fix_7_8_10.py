"""Security regressions; all service calls are replaced locally."""
import asyncio
import json
import shlex
from types import SimpleNamespace

import pytest
import yaml

from scripts import bench_asr, gemini_live_transcribe, probe_live
from src import settings_edit
from src.dispatch import DispatchConfig, _agent_command, dispatch
from src.llm.meeting_state import Item, MeetingState
from src.text import task_hub_dictionary


@pytest.mark.parametrize("value", ['a\\', 'a"b', 'a\\"', 'first\nsecond', '\nname\n',
                                  'a\r\t\x00\x1b\x85\u2028\u2029', 'true', '123', '利用者😀', 'a\x7f\x84\x9f'])
def test_settings_text_round_trip(tmp_path, value):
    path = tmp_path / 'settings.yaml'
    path.write_text('meeting:\n  self_name: old  # keep\n')
    assert settings_edit.set_value(path, 'meeting.self_name', value) == value
    assert yaml.safe_load(path.read_text())['meeting']['self_name'] == value
    assert '# keep' in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('speaker', ['a\\', 'a"b', 'a\\"', 'a\nb', 'true'])
def test_persona_round_trip_and_update(tmp_path, speaker):
    path = tmp_path / 'settings.yaml'
    path.write_text('task_hub:\n  personas:\n    old: old\n')
    assert settings_edit.set_persona(path, speaker, ['a\\', 'a"b'])[speaker] == ['a\\', 'a"b']
    assert settings_edit.set_persona(path, speaker, ['new'])[speaker] == ['new']
    assert len(yaml.safe_load(path.read_text())['task_hub']['personas']) == 2


@pytest.mark.parametrize('persona', [False, True])
def test_invalid_yaml_does_not_replace_original(tmp_path, monkeypatch, persona):
    path = tmp_path / 'settings.yaml'
    original = 'task_hub:\n  personas:\n    old: old\nmeeting:\n  self_name: old\n'
    path.write_text(original)
    monkeypatch.setattr(settings_edit, '_quote', lambda value: '"unterminated')
    with pytest.raises(yaml.YAMLError):
        if persona:
            settings_edit.set_persona(path, 'new', 'name')
        else:
            settings_edit.set_value(path, 'meeting.self_name', 'name')
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_valid_yaml_with_wrong_value_does_not_replace(tmp_path, monkeypatch):
    path = tmp_path / 'settings.yaml'
    original = 'meeting:\n  self_name: old\n  other: old\n'
    path.write_text(original)
    monkeypatch.setattr(settings_edit, '_line_of', lambda *args: 2)
    with pytest.raises(ValueError, match='一致'):
        settings_edit.set_value(path, 'meeting.self_name', 'new')
    assert path.read_text() == original


def test_replace_failure_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / 'settings.yaml'
    original = 'meeting:\n  self_name: old\n'
    path.write_text(original)
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(settings_edit.os, 'replace', fail)
    with pytest.raises(OSError):
        settings_edit.set_value(path, 'meeting.self_name', 'new')
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('template', [DispatchConfig().agent_command,
    "claude '{brief} を読み、このタスクに着手してください。完了したら {brief} の末尾に結果を追記してください'",
    'tool --brief {brief}'])
def test_brief_shell_quoting(tmp_path, template):
    path = tmp_path / "a ' $(touch INJECTED); `echo bad`\nbrief.md"
    command = _agent_command(template, path)
    if template.startswith('tool'):
        assert shlex.split(command) == ['tool', '--brief', str(path)]
    else:
        assert shlex.split(command) == ['claude', f'{path} を読み、このタスクに着手してください。完了したら {path} の末尾に結果を追記してください']


@pytest.mark.parametrize('template', ['tool {task}', "tool '{task}'", 'tool {task!r}',
                                     'tool {task.text}', "tool '{brief}'", 'tool "{brief}"'])
def test_unsafe_command_never_starts_agent(tmp_path, template):
    calls = []
    todo = Item('T1', '$(touch bad); malicious')
    agent = SimpleNamespace(create_terminal=lambda *args: calls.append(args))
    result = dispatch(tmp_path, MeetingState(todos=[todo]), todo, [], {},
                      DispatchConfig(agent_repo='/repo', agent_command=template), None, agent, 'owner')
    assert 'agent' in result.errors
    assert calls == []


def test_title_controls_are_removed(tmp_path):
    calls = []
    todo = Item('T1', 'normal\n\r\t\x1b\x00\x7f\u2028\u2029end')
    agent = SimpleNamespace(create_terminal=lambda *args: calls.append(args))
    dispatch(tmp_path, MeetingState(todos=[todo]), todo, [], {},
             DispatchConfig(agent_repo='/repo'), None, agent, 'owner')
    assert calls[0][1] == '▶ normalend'


@pytest.mark.parametrize('pair', [('--audit', 'good'), ('-word', 'good'), ('good', '--audit')])
def test_option_like_dictionary_words_are_skipped(tmp_path, monkeypatch, pair):
    (tmp_path / 'dict_correct.py').write_text('')
    calls = []
    def run(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout=json.dumps({'status': 'added'}))
    monkeypatch.setattr(task_hub_dictionary.subprocess, 'run', run)
    results = task_hub_dictionary.add_candidates(tmp_path, 'client', [pair, ('normal', 'correct')])
    assert results[0]['status'] == 'skipped'
    assert 'CLI' in results[0]['detail']
    assert results[1]['status'] == 'added'
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['recording_remote.wav', 'recording_self.123.wav',
                                 'recording_room.wav', 'meeting.mp3', 'audio/remote.mp3',
                                 'recording_remote.wav.age'])
def test_meeting_filenames_outside_sessions_are_refused(tmp_path, name):
    with pytest.raises(SystemExit, match='実会議の音声'):
        bench_asr.refuse_real_meetings(tmp_path / name, 'deepgram')
    bench_asr.refuse_real_meetings(tmp_path / name, 'local')


@pytest.mark.parametrize('marker', ['.encryption.json', 'transcripts.jsonl.age'])
def test_encrypted_meeting_contents_are_refused(tmp_path, marker):
    (tmp_path / marker).write_text('{}')
    audio = tmp_path / 'nested' / 'public.wav'
    with pytest.raises(SystemExit, match='実会議の音声'):
        bench_asr.refuse_real_meetings(audio, 'gemini')


def test_symlink_to_meeting_is_refused(tmp_path):
    target = tmp_path / 'recording_remote.wav'
    target.touch()
    link = tmp_path / 'public.wav'
    link.symlink_to(target)
    with pytest.raises(SystemExit):
        bench_asr.refuse_real_meetings(link, 'deepgram')


def test_deepgram_opts_out_of_training(tmp_path, monkeypatch):
    import httpx
    calls = []
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {'results': {'channels': [{'alternatives': [{'transcript': 'ok'}]}]}})
    monkeypatch.setattr(httpx, 'Client', Client)
    monkeypatch.setattr(bench_asr, 'read_key', lambda *args: 'fake-key')
    path = tmp_path / 'public.wav'
    path.write_bytes(b'fake audio')
    assert bench_asr.run_deepgram(path) == 'ok'
    assert calls[0]['params']['mip_opt_out'] == 'true'


@pytest.mark.parametrize('module', [probe_live, gemini_live_transcribe])
def test_websocket_header_and_redacted_receive_error(tmp_path, monkeypatch, capsys, module):
    import websockets
    key = 'FAKE-SECRET-KEY'
    calls = []
    class Socket:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def send(self, message): pass
        async def recv(self): return '{"setupComplete": {}}'
        def __aiter__(self): return self
        async def __anext__(self): raise RuntimeError('headers: ' + key)
    def connect(url, **kwargs):
        calls.append((url, kwargs))
        return Socket()
    original_sleep = asyncio.sleep
    async def sleep(*args): await original_sleep(0)
    monkeypatch.setattr(websockets, 'connect', connect)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    monkeypatch.setattr(module, 'load_pcm16', lambda *args: b'')
    if module is probe_live:
        asyncio.run(module.probe('model', key, tmp_path / 'fake.wav', 0, False))
    else:
        asyncio.run(module.one_session('model', key, b'', 0, 0, 'speaker', [], []))
    assert calls
    for url, kwargs in calls:
        assert '?' not in url and key not in url
        assert kwargs['additional_headers'] == {'x-goog-api-key': key}
    output = capsys.readouterr().out
    assert key not in output
    if module is probe_live:
        assert 'RuntimeError' in output and '[REDACTED]' in output


def test_probe_redacts_handshake_error(tmp_path, monkeypatch, capsys):
    import websockets
    def fail(*args, **kwargs): raise RuntimeError('secret in request: FAKE-KEY')
    monkeypatch.setattr(websockets, 'connect', fail)
    monkeypatch.setattr(probe_live, 'load_pcm16', lambda *args: b'')
    assert asyncio.run(probe_live.probe('model', 'FAKE-KEY', tmp_path / 'fake.wav', 0, False)) == 1
    output = capsys.readouterr().out
    assert 'FAKE-KEY' not in output
    assert 'RuntimeError' in output and '[REDACTED]' in output


def test_gemini_main_does_not_print_connection_secret(tmp_path, monkeypatch, capsys):
    import sys
    config = tmp_path / 'settings.yaml'
    config.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['gemini_live_transcribe.py', str(tmp_path / 'public.wav'),
        '--out', str(tmp_path / 'out.jsonl'), '--billing-project', 'fake-project',
        '--key-file', str(tmp_path / 'fake.key'), '--config', str(config)])
    monkeypatch.setattr(gemini_live_transcribe.ExternalSttConfig, 'from_mapping',
                        lambda *args: SimpleNamespace(billing_project='fake-project'))
    monkeypatch.setattr(gemini_live_transcribe, 'require_gemini_audio_key', lambda *args, **kwargs: 'FAKE-KEY')
    monkeypatch.setattr(gemini_live_transcribe, 'load_pcm16', lambda *args: b'')
    async def fail(*args): raise RuntimeError('headers: FAKE-KEY')
    monkeypatch.setattr(gemini_live_transcribe, 'stream', fail)
    assert gemini_live_transcribe.main() == 1
    captured = capsys.readouterr()
    assert 'FAKE-KEY' not in captured.out + captured.err
    assert 'RuntimeError' in captured.err



@pytest.mark.parametrize("template", ["claude `cat {brief}`", "claude $(cat {brief})"])
def test_agent_command_rejects_command_substitution(tmp_path, template):
    from src import dispatch
    with pytest.raises(ValueError):
        dispatch._agent_command(template, tmp_path / "brief.md")
