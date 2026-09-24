"""会議モードの配線の結合テスト。

実デバイス・Whisper・Ollama・resemblyzer は使わず、音声フレームを流し込んで
「2系統が混ざらないこと」「相手側だけ話者判定にかかること」「自分側は固定ラベルの
ままであること」を確認する。
"""

import json
import threading
import time
from datetime import datetime

import numpy as np
import pytest
from concurrent.futures import Future

from src.audio import multi_capture as mc
from src.audio.multi_capture import MultiAudioConfig, MultiCapture, SourceConfig
from src.audio.vad import AudioChunk
from src.llm.meeting_state import Item, MeetingState
from src.llm.state_updater import UpdateStats
from src import meeting_orchestrator as mo
from src.meeting_orchestrator import REMOTE_KEY, SELF_KEY, MeetingConfig, MeetingOrchestrator, detect_tone
from src.stt.whisper_client import TranscriptSegment

SR = 48000


class FakeStream:
    instances: list["FakeStream"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        FakeStream.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def emit(self, frames):
        self.callback(frames, len(frames), None, None)


class FakeDiarizer:
    """振幅で話者を決める偽ダイアライザ（0.5 → 田中 / それ以外 → 不明話者1）。"""

    def __init__(self):
        self.calls = 0
        self.unknown_names = []
        self.dissolved_names = []
        self.enrolled_names = ["田中", "佐藤"]
        from src.audio.enrolled_diarizer import DiarizerConfig

        self.config = DiarizerConfig()

    def embed(self, audio, sample_rate):
        return np.array([float(np.max(np.abs(audio)))], dtype=np.float32)

    def identify_embedding(self, embedding, duration_sec, at):
        self.calls += 1
        from src.audio.enrolled_diarizer import IdentifyResult

        if float(embedding[0]) == pytest.approx(0.5, abs=0.01):
            return IdentifyResult(name="田中", similarity=0.9)
        return IdentifyResult(name="不明話者1", similarity=0.2, is_new=True, confident=False)

    def partial_embeddings(self, audio, sample_rate, rate=2.5):
        return []

    def report_text(self, name, chars):
        pass

    def rename(self, old, new):
        if old not in self.enrolled_names:
            raise KeyError(old)
        self.enrolled_names[self.enrolled_names.index(old)] = new

    def scores(self, embedding):
        return {"田中": float(embedding[0]) + 0.4}

    def mark_named(self, name):
        pass

    def add_anchor(self, name, embedding):
        pass

    def merge_unknowns(self):
        return []

    def __len__(self):
        return 1


def _config(tmp_path) -> dict:
    return {
        "audio": {"sample_rate": SR, "chunk_duration_sec": 0.1},
        "vad": {
            "silence_threshold_db": -40,
            "silence_duration_sec": 0.3,
            "min_chunk_sec": 0.2,
            "max_chunk_sec": 10,
        },
        "stt": {},
        "llm": {},
        "output": {"workspace_dir": str(tmp_path)},
        "orchestrator": {"summary_interval_sec": 999},
        "ui": {"enabled": False},
        "meeting": {
            "capture_backend": "blackhole",
            "self_name": "自分",
            "mic_candidates": ["Headset One"],
            "first_frame_timeout_sec": 0.05,   # テストでは実フレームが来ないので短くする
        },
    }


@pytest.fixture
def orch(tmp_path, monkeypatch):
    FakeStream.instances = []
    monkeypatch.setattr(mc.sd, "InputStream", FakeStream)

    o = MeetingOrchestrator(_config(tmp_path), tmp_path)
    o.mic = type("D", (), {"index": 6, "name": "Headset One", "channels": 1})()
    o.loopback = type("D", (), {"index": 8, "name": "BlackHole 2ch", "channels": 2})()
    o.diarizer = FakeDiarizer()

    # STT は「話者名をそのまま返す」スタブ（実際の話者ラベル付けを検証するため）
    def fake_transcribe(chunk: AudioChunk):
        return TranscriptSegment(
            speaker=chunk.speaker,
            text=f"{chunk.speaker} の発話",
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            timestamp="2026-09-08T12:00:00",
        )

    monkeypatch.setattr(o.whisper, "transcribe", fake_transcribe)
    return o


def _speech(seconds: float, amplitude: float, channels: int) -> np.ndarray:
    """VAD が発話と判定するレベルのフレーム。"""
    frames = int(seconds * SR)
    return np.full((frames, channels), amplitude, dtype=np.float32)


def _silence(seconds: float, channels: int) -> np.ndarray:
    return np.zeros((int(seconds * SR), channels), dtype=np.float32)


class TestConfig:
    def test_mic_candidates_come_from_settings(self, tmp_path):
        o = MeetingOrchestrator(_config(tmp_path), tmp_path)
        assert o.meeting.mic_candidates == ["Headset One"]
        assert o.meeting.self_name == "自分"

    def test_defaults_cover_the_candidates_case(self):
        cfg = MeetingConfig(mic_candidates=["Headset One", "Shure MV7", "AirPods Pro"])
        assert cfg.loopback_candidates == ["BlackHole 2ch"]
        assert cfg.multi_output_name == "会議録音用（複数出力装置）"

    def test_listening_device_defaults_to_the_mic_itself(self):
        """ヘッドセットはマイクと出力が同名なので、書かなければそのまま使う。"""
        cfg = MeetingConfig(mic_candidates=["Headset One"])
        assert cfg.listening_device_for.get("Headset One", "Headset One") == "Headset One"

    def test_listening_device_can_differ_from_the_mic(self):
        """据え置きマイクは聴く側が別。束に要るのは聴く側の方。"""
        cfg = MeetingConfig(
            mic_candidates=["Shure MV7"],
            listening_device_for={"Shure MV7": "USB Audio Device"},
        )
        assert cfg.listening_device_for["Shure MV7"] == "USB Audio Device"

    def test_listening_map_is_read_from_config(self, tmp_path):
        config = _config(tmp_path)
        config["meeting"]["listening_device_for"] = {"Shure MV7": "USB Audio Device"}
        o = MeetingOrchestrator(config, tmp_path)
        assert o.meeting.listening_device_for == {"Shure MV7": "USB Audio Device"}

    def test_writer_headings_are_switched_to_meeting(self, tmp_path):
        o = MeetingOrchestrator(_config(tmp_path), tmp_path)
        assert o.writer.config.title == "会議"
        assert o.writer.config.items_label == "前回タスク 消化状況"

    def test_meeting_prompt_is_preferred(self, tmp_path):
        o = MeetingOrchestrator(_config(tmp_path), tmp_path)
        o.load_prep_materials(tmp_path / "does-not-exist")
        assert "会議" in o._system_prompt

    def test_rolling_prompt_is_preferred(self, tmp_path):
        o = MeetingOrchestrator(_config(tmp_path), tmp_path)
        o.load_prep_materials(tmp_path / "does-not-exist")
        assert "ローリング状態" in o._system_prompt

    def test_session_info_includes_self_name_and_enrolled(self, orch):
        info = orch.session_info()
        assert info["self_name"] == "自分"
        assert info["enrolled"] == ["田中", "佐藤"]
        assert info["models"]["stt"]["where"] == "ローカル（mlx）"

    def test_session_info_includes_state_interval(self, orch):
        """画面が「次の更新（N 以内）」を出すのに使う。

        09-18 に要約を 30 秒 → 2 分へ変えたのに、画面は「30 秒以内」と直書きのままで、
        **使う人に嘘を言い続けていた**（2026-09-21 にスクショを撮り直そうとして気づいた）。
        """
        assert orch.session_info()["state_interval_sec"] == orch._state_cfg.interval_sec

    def test_dispatch_finds_todo_and_uses_self_for_unassigned_owner(self, orch, monkeypatch):
        """未定 TODO の払い出しは本人を担当にして dispatch へ渡す。"""
        state = MeetingState(todos=[Item("T1", "確認する", by="未定", due="未定")])
        orch._state_loop = type("State", (), {"state": lambda self: state})()
        captured = {}

        def fake_dispatch(*args):
            captured["todo"] = args[2]
            captured["owner"] = args[-1]
            return __import__("src.dispatch", fromlist=["DispatchResult"]).DispatchResult("T1", orch.session_dir / "dispatch" / "T1.md")

        monkeypatch.setattr("src.meeting_orchestrator.dispatch", fake_dispatch)
        assert orch.dispatch("T1")["todo_id"] == "T1"
        assert captured["todo"].id == "T1"
        assert captured["owner"] == "自分"

    def test_dispatch_unknown_todo_raises_key_error(self, orch):
        """存在しない TODO は UI が 404 にできる KeyError にする。"""
        orch._state_loop = type("State", (), {"state": lambda self: MeetingState()})()
        with pytest.raises(KeyError):
            orch.dispatch("T404")

    def test_missing_task_hub_shared_directory_does_not_stop_startup(self, tmp_path):
        """共有 タスク管理 モジュールが無くても、会議本体は Notion 無効で作れる。"""
        config = _config(tmp_path)
        config["task_hub"] = {"shared_dir": str(tmp_path / "missing")}
        orchestrator = MeetingOrchestrator(config, tmp_path)
        assert orchestrator._bridge is None

    def test_participants_returns_enrolled_speaker_and_utterance_count(self, orch):
        """参加者一覧は登録話者と文字起こし済み発話数を返す。"""
        future = Future()
        future.set_result([TranscriptSegment("田中", "確認します", 1.0, 2.0, "")])
        orch._on_transcribed(future)
        participants = {item["name"]: item for item in orch.participants()}
        assert participants["田中"] == {"name": "田中", "enrolled": True, "utterances": 1, "chars": 5, "suggestions": []}

    def test_rename_speaker_calls_diarizer(self, orch):
        orch.rename_speaker("田中", "鈴木")
        assert orch.diarizer.enrolled_names[0] == "鈴木"

    def test_relabel_segment_records_correction_and_updates_buffer(self, orch, tmp_path):
        """行単位の訂正は JSONL に残し、未処理バッファへ反映する。"""
        from src.stt.whisper_client import TranscriptSegment

        orch._transcript_buffer.append(TranscriptSegment("不明話者?", "発話", 1.0, 2.0, ""))
        orch.relabel_segment(1.03, 2.0, "不明話者?", "田中")
        records = (tmp_path / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(records) == 1
        assert json.loads(records[0])["new"] == "田中"
        assert orch._transcript_buffer[0].speaker == "田中"

    def test_ghost_participant_rename_updates_alias_and_stats_without_diarizer(self, orch):
        """一行の付け替えで生じた diarizer 外の名前も安全に改名できる。"""
        orch._speaker_stats["とりのこ"] = [1, 4]
        orch.rename_speaker("とりのこ", "鳥の子")
        assert orch._aliases["とりのこ"] == "鳥の子"
        assert orch._speaker_stats == {"鳥の子": [1, 4]}
        assert orch.diarizer.enrolled_names == ["田中", "佐藤"]

    def test_relabel_label_is_preserved_and_renamed(self, orch):
        """行単位の付け替えは表示ラベルにのみ残り、改名時に追従する。"""
        orch.relabel_segment(1.001, 2.0, "不明話者?", "とりのこ")
        assert orch._row_labels == {1.0: "とりのこ"}
        orch.rename_speaker("とりのこ", "鳥の子")
        assert orch._row_labels == {1.0: "鳥の子"}

    def test_participants_hides_empty_ghost_participant(self, orch):
        """発話を全て移した diarizer 外の名前は参加者一覧に残さない。"""
        orch._speaker_stats["とりのこ"] = [0, 0]
        assert "とりのこ" not in {item["name"] for item in orch.participants()}


class TestCaptureWiring:
    def test_two_streams_are_opened(self, orch):
        orch._start_capture()
        assert len(FakeStream.instances) == 2
        assert [s.kwargs["device"] for s in FakeStream.instances] == [6, 8]

    def test_builders_get_the_start_offsets(self, orch, monkeypatch):
        clock = iter([500.0, 500.4])
        monkeypatch.setattr(mc.time, "time", lambda: next(clock))

        capture = MultiCapture(
            [
                SourceConfig(SELF_KEY, 6, "mic", 1, "自分"),
                SourceConfig(REMOTE_KEY, 8, "BlackHole 2ch", 2),
            ],
            MultiAudioConfig(sample_rate=SR),
        )
        FakeStream.instances[0].emit(_silence(0.1, 1))
        FakeStream.instances[1].emit(_silence(0.1, 2))

        offsets = capture.start_offsets()
        assert offsets[SELF_KEY] == pytest.approx(0.0)
        assert offsets[REMOTE_KEY] == pytest.approx(0.4)


class TestSpeakerLabelling:
    def _run_one_cycle(self, orch):
        orch._start_capture()
        mic, loopback = FakeStream.instances

        # 自分が 0.5 秒話して黙る
        mic.emit(_speech(0.5, 0.3, 1))
        mic.emit(_silence(0.5, 1))
        # 相手（田中さん）が 0.5 秒話して黙る
        loopback.emit(_speech(0.5, 0.5, 2))
        loopback.emit(_silence(0.5, 2))

        orch._process_audio_queues()
        orch._executor.shutdown(wait=True)

    def test_self_keeps_the_fixed_label(self, orch):
        self._run_one_cycle(orch)
        speakers = [s.speaker for s in orch._transcript_buffer]
        assert "自分" in speakers

    def test_remote_is_labelled_by_the_diarizer(self, orch):
        self._run_one_cycle(orch)
        speakers = [s.speaker for s in orch._transcript_buffer]
        assert "田中" in speakers

    def test_diarizer_is_not_run_on_own_voice(self, orch):
        """自分側はマイクを分けているので判定しない（判定すると誤ラベルの元になる）。"""
        self._run_one_cycle(orch)
        assert orch.diarizer.calls == 1  # 相手側の1チャンクだけ

    def test_both_sources_produce_segments(self, orch):
        self._run_one_cycle(orch)
        assert len(orch._transcript_buffer) == 2

    def test_transcript_jsonl_is_written(self, orch, tmp_path):
        self._run_one_cycle(orch)
        lines = (tmp_path / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert any("田中" in line for line in lines)

    def test_segment_is_published(self, orch):
        future = Future()
        future.set_result([TranscriptSegment("田中", "確認します", 1.0, 2.0, "")])
        orch._on_transcribed(future)
        event = orch.bus.subscribe().get(0.1)
        assert event.type == "segment"
        assert event.data["text"] == "確認します"

    def test_remote_chunk_with_two_turns_creates_two_segments(self, orch, monkeypatch):
        from src.audio.turn_splitter import Turn

        chunk = AudioChunk("remote", np.ones(SR * 4, dtype=np.float32), 3.0, 7.0, SR)
        turns = [
            Turn(0.0, 2.0, "田中", 0.9, True, np.array([0.5], dtype=np.float32)),
            Turn(2.0, 4.0, "佐藤", 0.8, True, np.array([0.4], dtype=np.float32)),
        ]
        monkeypatch.setattr("src.meeting_orchestrator.split_turns", lambda *args: turns)

        segments = orch._split_and_transcribe(chunk)

        assert [segment.speaker for segment in segments] == ["田中", "佐藤"]

    def test_initial_prompt_has_terms_but_not_speaker_names(self, orch, tmp_path):
        """2026-09-11 本番: 登録名をヒントに入れると、文中に「GX」が紛れ込み、無音区間でヒントを読み上げた。"""
        prep = tmp_path / "prep"
        prep.mkdir()
        (prep / "terms.md").write_text("## 用語\n- 固有名詞\n", encoding="utf-8")
        orch.load_prep_materials(prep)

        prompt = orch._build_initial_prompt()

        assert prompt == "固有名詞"
        assert "田中" not in prompt


class TestTranscriptFormatting:
    def test_uses_real_speaker_names(self, orch):
        segments = [
            TranscriptSegment("自分", "前回の宿題ですが", 0.0, 2.0, ""),
            TranscriptSegment("田中", "こちらは完了しました", 2.5, 5.0, ""),
            TranscriptSegment("不明話者1", "補足させてください", 5.5, 8.0, ""),
        ]
        text = orch._format_transcript(segments)
        assert "[0:00] 自分: 前回の宿題ですが" in text
        assert "[0:02] 田中: こちらは完了しました" in text
        assert "[0:05] 不明話者1: 補足させてください" in text

    def test_sorts_by_time(self, orch):
        segments = [
            TranscriptSegment("田中", "あとの発言", 10.0, 12.0, ""),
            TranscriptSegment("自分", "さきの発言", 1.0, 2.0, ""),
        ]
        lines = orch._format_transcript(segments).splitlines()
        assert "さきの発言" in lines[0]

    def test_formats_hours(self, orch):
        assert MeetingOrchestrator._format_seconds(3725) == "1:02:05"
        assert MeetingOrchestrator._format_seconds(65) == "1:05"


class TestRollingState:
    def test_transcribed_segment_is_pushed_to_state_loop(self, orch):
        class FakeStateLoop:
            def __init__(self):
                self.segments: list[TranscriptSegment] = []

            def push(self, segment: TranscriptSegment) -> None:
                self.segments.append(segment)

        state_loop = FakeStateLoop()
        orch._state_loop = state_loop
        future = Future()
        segment = TranscriptSegment("自分", "確認します", 3.0, 4.0, "")
        future.set_result([segment])

        orch._on_transcribed(future)

        assert state_loop.segments == [segment]

    def test_on_state_writes_state_and_changes(self, orch, tmp_path):
        state = MeetingState(
            updated_at=65.0,
            summary="決定を記録しました。",
            decisions=[Item("D1", "実装する", by="自分")],
        )

        orch._on_state(state, UpdateStats(prompt_chars=123, latency_sec=1.5), ["+ [D1] 実装する（open）"])

        assert '"D1"' in (tmp_path / "state.json").read_text(encoding="utf-8")
        assert "決定事項" in (tmp_path / "interview_summary.md").read_text(encoding="utf-8")
        assert "+ [D1] 実装する（open）" in (tmp_path / "interview_log.md").read_text(encoding="utf-8")

        event = orch.bus.subscribe().get(0.1)
        assert event.type == "state"
        assert event.data["latency_sec"] == 1.5

    def test_level_is_published_once_per_second(self, orch, monkeypatch):
        class Capture:
            def drain(self, key):
                return [_speech(0.1, 0.1 if key == SELF_KEY else 0.2, 1)]

        orch.capture = Capture()
        orch._builders = {SELF_KEY: type("Builder", (), {"feed": lambda self, frame: []})(), REMOTE_KEY: type("Builder", (), {"feed": lambda self, frame: []})()}
        monkeypatch.setattr("src.meeting_orchestrator.time.time", lambda: 10.0)
        subscription = orch.bus.subscribe()

        orch._last_level_at = 8.9
        orch._process_audio_queues()
        orch._last_level_at = 9.5
        orch._process_audio_queues()
        orch._last_level_at = 8.9
        orch._process_audio_queues()

        events = []
        while (event := subscription.get(0.01)) is not None:
            events.append(event)
        assert len([event for event in events if event.type == "level"]) == 2

    def test_legacy_uses_summary_cycle(self, orch, monkeypatch):
        orch.meeting.summary_mode = "legacy"
        orch.meeting.skip_enrollment = True
        calls: list[str] = []

        def fake_process() -> int:
            if calls:
                raise KeyboardInterrupt
            with orch._buffer_lock:
                orch._transcript_buffer.append(TranscriptSegment("自分", "発話", 0.0, 1.0, ""))
            return 0

        monkeypatch.setattr(orch.llm, "health_check", lambda: True)
        monkeypatch.setattr("src.meeting_orchestrator.dev.resolve_input_device", lambda *args, **kwargs: object())
        monkeypatch.setattr(orch.writer, "init_files", lambda *args: None)
        monkeypatch.setattr(orch.whisper, "_ensure_model", lambda: None)
        monkeypatch.setattr(orch, "_preflight", lambda: True)
        monkeypatch.setattr(orch, "_build_initial_prompt", lambda: None)
        monkeypatch.setattr(orch, "_start_capture", lambda: None)
        monkeypatch.setattr(orch, "_process_audio_queues", fake_process)
        monkeypatch.setattr(orch, "_shutdown", lambda *args: None)
        monkeypatch.setattr(orch, "_run_summary_cycle", lambda: calls.append("summary"))
        monkeypatch.setattr("src.enrollment.load_or_create", lambda *args, **kwargs: (FakeDiarizer(), False))
        orch._summary_interval = -1

        orch.run()

        assert calls == ["summary"]

    def test_disabled_ui_does_not_start_server(self, orch, monkeypatch):
        orch.meeting.summary_mode = "legacy"
        orch.meeting.skip_enrollment = True
        started: list[bool] = []
        monkeypatch.setattr(orch.llm, "health_check", lambda: True)
        monkeypatch.setattr("src.meeting_orchestrator.dev.resolve_input_device", lambda *args, **kwargs: object())
        monkeypatch.setattr(orch.writer, "init_files", lambda *args: None)
        monkeypatch.setattr(orch.whisper, "_ensure_model", lambda: None)
        monkeypatch.setattr(orch, "_preflight", lambda: True)
        monkeypatch.setattr(orch, "_build_initial_prompt", lambda: None)
        monkeypatch.setattr(orch, "_start_capture", lambda: None)
        monkeypatch.setattr(orch, "_shutdown", lambda *args: None)
        monkeypatch.setattr(orch, "_process_audio_queues", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
        monkeypatch.setattr(orch, "_run_summary_cycle", lambda: None)
        monkeypatch.setattr("src.enrollment.load_or_create", lambda *args, **kwargs: (FakeDiarizer(), False))
        monkeypatch.setattr("src.meeting_orchestrator.start_ui_server", lambda *args: started.append(True))

        orch.run()

        assert not started


class TestDiarizerFailureIsNotFatal:
    def test_identify_error_falls_back_to_a_label(self, orch):
        class Broken:
            def identify(self, audio, sr):
                raise RuntimeError("encoder died")

        orch.diarizer = Broken()
        chunk = AudioChunk("remote", np.full(24000, 0.5, dtype=np.float32), 0.0, 0.5, SR)
        segments = orch._split_and_transcribe(chunk)
        assert len(segments) == 1
        assert segments[0].speaker == "不明話者?"


class TestRenameAliases:
    """改名の瞬間に文字起こし中だった発話が旧ラベルで届き、改名前後の話者が両方残る（運用者 目視 2026-09-09）。"""

    def test_late_segment_with_old_label_is_renamed_on_arrival(self, orch):
        from concurrent.futures import Future

        orch.rename_speaker("田中", "TANAKA")
        future = Future()
        future.set_result([TranscriptSegment("田中", "改名の瞬間に処理中だった発話", 10.0, 12.0, "")])
        orch._on_transcribed(future, submitted_at=None)
        assert orch._transcript_buffer[-1].speaker == "TANAKA"
        assert [p["name"] for p in orch.participants() if p["utterances"]] == ["TANAKA"]

    def test_rename_back_and_forth_does_not_leave_both(self, orch):
        from concurrent.futures import Future

        orch.rename_speaker("田中", "TANAKA")
        orch.rename_speaker("TANAKA", "田中")
        future = Future()
        future.set_result([TranscriptSegment("TANAKA", "遅れて届いた", 10.0, 12.0, "")])
        orch._on_transcribed(future, submitted_at=None)
        names = [p["name"] for p in orch.participants() if p["utterances"]]
        assert names == ["田中"]

    def test_relabel_moves_one_utterance_between_participants(self, orch, tmp_path):
        from concurrent.futures import Future

        future = Future()
        future.set_result([TranscriptSegment("田中", "abcde", 10.0, 12.0, "")])
        orch._on_transcribed(future, submitted_at=None)
        orch.relabel_segment(10.0, 12.0, "田中", "佐藤")
        stats = {p["name"]: (p["utterances"], p["chars"]) for p in orch.participants()}
        assert stats["佐藤"] == (1, 5)
        assert stats.get("田中", (0, 0)) == (0, 0)


def _tone(total_sec: float, hz: float = 880.0, dbfs: float = -20.0, start: float = 0.3, sec: float = 0.2) -> np.ndarray:
    """total_sec の無音の中に、start 秒から sec 秒だけ純音を置く。"""
    out = np.zeros(int(total_sec * SR), dtype=np.float32)
    t = np.arange(int(sec * SR)) / SR
    i = int(start * SR)
    out[i : i + t.size] = 10 ** (dbfs / 20) * np.sin(2 * np.pi * hz * t)
    return out


def _voice(total_sec: float, f0: float = 132.0, peak_dbfs: float = -13.0) -> np.ndarray:
    """抑揚と音節のある倍音列（声の代用品）。"""
    t = np.arange(int(total_sec * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 30))
    sig *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    return (sig / np.max(np.abs(sig)) * 10 ** (peak_dbfs / 20)).astype(np.float32)


class TestDetectTone:
    def test_beep_alone_is_found(self):
        assert detect_tone(_tone(1.5), SR, 880).tone_sec >= 0.1

    def test_beep_is_found_while_the_other_side_is_talking(self):
        # 2026-09-11 の実機: 0.2 秒のビープが 264 Hz の声に負けて「SCK に届いていない」と誤報した
        result = detect_tone(_tone(1.5) + _voice(1.5), SR, 880)
        assert result.dominant_hz != pytest.approx(880, abs=10)   # 旧判定（いちばん強い周波数）なら落ちる
        assert result.tone_sec >= 0.1

    def test_voice_alone_is_not_a_beep(self):
        assert detect_tone(_voice(1.5), SR, 880).tone_sec == 0.0

    def test_silence_and_constant_lsb_are_not_a_beep(self):
        assert detect_tone(np.zeros(int(1.5 * SR), dtype=np.float32), SR, 880).tone_sec == 0.0
        # Headset One が無言のときに返す値（−90.3 dBFS 固定）
        assert detect_tone(np.full(int(1.5 * SR), 1 / 32768, dtype=np.float32), SR, 880).tone_sec == 0.0

    def test_other_frequency_is_not_the_beep(self):
        assert detect_tone(_tone(1.5, hz=440), SR, 880).tone_sec == 0.0

    def test_column_shaped_frames_are_accepted(self):
        assert detect_tone(_tone(1.5).reshape(-1, 1), SR, 880).tone_sec >= 0.1

    def test_empty_recording(self):
        assert detect_tone(np.empty(0, dtype=np.float32), SR, 880).tone_sec == 0.0


class TestPreflight:
    @pytest.fixture
    def sck(self, orch, monkeypatch):
        orch.meeting.capture_backend = "screencapturekit"
        orch.meeting.level_check_sec = 0.5
        orch.loopback = None
        orch.mic.matched_candidate = "Headset One"
        monkeypatch.setattr(mo, "responsible_app_name", lambda: "Terminal")
        # 測定はボタン（画面）か Enter（端末）で始まる。端末経路では Enter＝測る、最後の [y/N] は n
        asked: list[str] = []

        def fake_input(prompt: str = "") -> str:
            asked.append(prompt)
            return "" if "準備ができたら" in prompt else "n"

        monkeypatch.setattr("builtins.input", fake_input)

        def wire(self_audio: np.ndarray, remote_audio: np.ndarray, beep_ok: bool) -> list[str]:
            sources = {SELF_KEY: self_audio, REMOTE_KEY: remote_audio}
            orch.capture = type("C", (), {"record_from": lambda _self, key, seconds: sources[key]})()
            monkeypatch.setattr(orch, "_beep_selftest", lambda: beep_ok)
            return asked

        return wire

    def test_asks_the_user_to_speak_before_measuring_the_mic(self, orch, sck, capsys):
        sck(_voice(0.5), _voice(0.5), beep_ok=True)
        assert orch._preflight() is True
        out = capsys.readouterr().out
        assert out.index("話してください") < out.index("自分のマイク : peak")

    def test_beep_miss_is_not_warned_when_the_other_side_is_audible(self, orch, sck, capsys):
        asked = sck(_voice(0.5), _voice(0.5), beep_ok=False)
        assert orch._preflight() is True
        assert all("開始しますか" not in prompt for prompt in asked)   # 警告が無いので確認もされない
        assert "画面収録の許可先" not in capsys.readouterr().out

    def test_beep_miss_with_silent_remote_is_warned(self, orch, sck, capsys):
        asked = sck(_voice(0.5), np.full(int(0.5 * SR), 1 / 32768, dtype=np.float32), beep_ok=False)
        assert orch._preflight() is False
        assert any("開始しますか" in prompt for prompt in asked)
        assert "画面収録の許可先『Terminal』" in capsys.readouterr().out

    def test_measurement_waits_for_the_go_ahead(self, orch, sck):
        """測定はボタン（端末なら Enter）を押してから始める。自動で始めない。"""
        asked = sck(_voice(0.5), _voice(0.5), beep_ok=True)
        orch._preflight()
        assert sum("準備ができたら" in prompt for prompt in asked) == 2   # マイクと相手側で 1 回ずつ

    def test_empty_capture_is_reported_as_a_device_problem(self, orch, sck, capsys):
        """1フレームも来ないのは「無音」ではなく機器・許可の問題として出す（09-12 実機で -100 dBFS）。"""
        sck(np.empty(0, dtype=np.float32), _voice(0.5), beep_ok=True)
        assert orch._preflight() is False
        assert "音声が1フレームも届いていません" in capsys.readouterr().out

    def test_silent_mic_is_warned(self, orch, sck, capsys):
        sck(np.full(int(0.5 * SR), 1 / 32768, dtype=np.float32), _voice(0.5), beep_ok=True)
        assert orch._preflight() is False
        assert "自分のマイク（Headset One）から音が来ていません" in capsys.readouterr().out


def _voice_vec(axis: int, seed: int, noise: float = 0.25) -> np.ndarray:
    """axis 方向を中心にした、同じ人らしい embedding。"""
    rng = np.random.default_rng(seed)
    base = np.zeros(256, dtype=np.float32)
    base[axis] = 1.0
    vec = base + noise * rng.standard_normal(256).astype(np.float32) / np.sqrt(256)
    return vec / np.linalg.norm(vec)


class TestLearnFromCorrections:
    """2026-09-11 本番: 行を直しても声を覚えず、直した直後からまた不明話者に戻っていた。"""

    @pytest.fixture
    def live(self, orch):
        from src.audio.enrolled_diarizer import EnrolledDiarizer

        orch.diarizer = EnrolledDiarizer()
        events: list[tuple[str, dict]] = []
        orch.bus.publish = lambda kind, data: events.append((kind, data))

        def row(start: float, seconds: float, axis: int, seed: int, speaker: str = "不明話者?"):
            orch._remember_voice(TranscriptSegment(speaker, "発話", start, start + seconds, ""), _voice_vec(axis, seed))

        return orch, row, events

    def test_correcting_one_row_names_the_matching_unassigned_rows(self, live):
        orch, row, events = live
        row(10.0, 3.0, axis=0, seed=1)    # 参加者A
        row(20.0, 1.0, axis=0, seed=2)    # 参加者A（短い）
        row(30.0, 2.0, axis=1, seed=3)    # 別人
        orch.relabel_segment(10.0, 13.0, "不明話者?", "参加者A")
        relabeled = [(data["start_time"], data["new"]) for kind, data in events if kind == "relabel"]
        assert relabeled == [(20.0, "参加者A")]
        assert "参加者A" in orch.diarizer.enrolled_names

    def test_later_short_utterances_now_match_the_named_speaker(self, live):
        orch, row, _ = live
        row(10.0, 3.0, axis=0, seed=1)
        orch.relabel_segment(10.0, 13.0, "不明話者?", "参加者A")
        result = orch.diarizer.identify_embedding(_voice_vec(0, seed=9), duration_sec=0.8, at=40.0)
        assert result.name == "参加者A"

    def test_short_row_only_changes_its_label(self, live):
        orch, row, events = live
        row(10.0, 0.8, axis=0, seed=1)
        row(20.0, 3.0, axis=0, seed=2)
        orch.relabel_segment(10.0, 10.8, "不明話者?", "参加者A")
        assert orch.diarizer.enrolled_names == []
        assert [kind for kind, _ in events if kind == "relabel"] == []

    def test_manually_corrected_rows_are_not_overwritten(self, live):
        orch, row, events = live
        row(10.0, 3.0, axis=0, seed=1)
        row(20.0, 3.0, axis=0, seed=2)
        orch.relabel_segment(20.0, 23.0, "参加者A", "不明話者?")   # 人が「この行は違う」と戻した
        orch.relabel_segment(10.0, 13.0, "不明話者?", "参加者A")
        assert [data["start_time"] for kind, data in events if kind == "relabel"] == []

    def test_renaming_an_unknown_speaker_makes_it_named(self, live):
        orch, _, _ = live
        orch.diarizer._speakers["不明話者1"] = __import__("src.audio.enrolled_diarizer", fromlist=["Speaker"]).Speaker(
            name="不明話者1", enrolled=False, enroll_embeddings=[_voice_vec(0, seed=1)]
        )
        orch.rename_speaker("不明話者1", "参加者A")
        assert "参加者A" in orch.diarizer.enrolled_names
        # 登録済みになったので、短い発話も付く
        assert orch.diarizer.identify_embedding(_voice_vec(0, seed=5), duration_sec=0.8, at=1.0).name == "参加者A"


class TestTranscriptTimesMatchTheRecording:
    """2026-09-11 本番: 登録の 71 秒（1 回目は開始確認の 502 秒）ぶん、文字起こしの時刻が録音より手前にずれていた。"""

    def test_audio_discarded_before_the_meeting_advances_the_clock(self, orch):
        orch._start_capture()
        mic, loopback = FakeStream.instances
        # セルフチェック・登録の 2 秒（キューから取り出して捨てる）
        mic.emit(_silence(2.0, 1))
        loopback.emit(_silence(2.0, 2))
        orch.capture.drain(SELF_KEY)
        orch.capture.drain(REMOTE_KEY)
        orch._align_clocks_to_recording()
        # 本編の最初の発話は、録音の 2 秒目から
        mic.emit(_speech(0.5, 0.3, 1))
        mic.emit(_silence(0.5, 1))
        orch._process_audio_queues()
        orch._executor.shutdown(wait=True)
        first = min(orch._transcript_buffer, key=lambda s: s.start_time)
        assert first.start_time == pytest.approx(2.0, abs=0.35)


def test_meeting_mode_turns_on_the_low_confidence_filter(orch):
    assert orch.whisper.config.min_avg_logprob == -1.0


class TestThroughputLog:
    def test_logs_how_far_behind_the_processing_is(self, orch, caplog):
        orch._stt_work.extend([(1.0, 2.0), (3.0, 3.0)])
        orch._last_state_latency_sec = 24.0
        with caplog.at_level("INFO"):
            orch._log_throughput(orch._started_at + 120)
        message = caplog.text
        assert "1 件 2.00 秒" in message
        assert "音声の 1.2 倍速" in message
        assert "要約 24 秒" in message

    def test_nothing_to_log_before_the_first_chunk(self, orch, caplog):
        with caplog.at_level("INFO"):
            orch._log_throughput(orch._started_at)
        assert "処理状況" not in caplog.text

    def test_work_time_is_recorded_per_chunk(self, orch):
        self_chunk = AudioChunk("自分", np.zeros(SR, dtype=np.float32), 5.0, 6.0, SR)
        orch._submit(SELF_KEY, self_chunk)
        orch._executor.shutdown(wait=True)
        assert len(orch._stt_work) == 1
        assert orch._stt_work[0][1] == pytest.approx(1.0)


def test_glossary_is_applied_to_transcribed_text(orch, tmp_path):
    """2026-09-11 本番の 1,644 行のうち 16 行が直る（「お母さん」→「自分さん」など）。"""
    orch._glossary = [("お母さん", "自分さん")]
    future = Future()
    future.set_result([TranscriptSegment("参加者A", "それをお母さんにパスして", 10.0, 12.0, "")])
    orch._on_transcribed(future)
    assert orch._transcript_buffer[-1].text == "それを自分さんにパスして"


class TestFactCheckWiring:
    def test_disabled_by_default(self, orch):
        assert orch._factcheck_cfg.enabled is False
        assert orch._factcheck_loop is None

    def test_claims_are_published_and_saved(self, orch, tmp_path):
        from src.llm.factcheck import Claim

        orch._on_claim(Claim("C1", "流入が 40% 下がった", "stat", "参加者A", 120.0, "要確認", "2024 年の数字とは違う"))
        event = orch.bus.subscribe().get(0.1)
        assert event.type == "factcheck"
        saved = json.loads((tmp_path / "factcheck.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert saved["verdict"] == "要確認"

    def test_settings_turn_it_on(self, tmp_path):
        config = _config(tmp_path)
        config["meeting"]["factcheck"] = {"enabled": True, "verify": "off"}
        o = MeetingOrchestrator(config, tmp_path)
        assert o._factcheck_cfg.enabled is True and o._factcheck_cfg.verify == "off"


class TestMidMeetingReview:
    """会議の途中で全文を読み直す（運用者 案 2026-09-12「終了 10 分前に一度まわす」）。"""

    @pytest.fixture
    def ready(self, orch, monkeypatch):
        from src.llm.meeting_state import Item, MeetingState

        state = MeetingState(summary="これまで")
        # 本物の StateLoop と同じく、state() は「いまの _state」を返す（見直しが差し替える）
        loop = type("Loop", (), {
            "state": lambda self: self._state,
            "_lock": threading.Lock(),
            "_state": state,
        })()
        orch._state_loop = loop
        monkeypatch.setattr(orch, "_read_transcripts", lambda: [TranscriptSegment("参加者A", "台湾の LP は 9 月中に出します", 10.0, 13.0, "")])
        return orch, Item

    def _fake_client(self, monkeypatch, available=True, decisions=("台湾向け LP を 9 月中に公開する",)):
        from src.llm.meeting_state import StateDelta

        class FakeClaude:
            def available(self_inner):
                return available

            def final_pass(self_inner, state, transcript, prompt):
                return StateDelta(summary="見直し後", current_topic="", topic_changed=False,
                                  new_decisions=[{"text": text, "by": "参加者A"} for text in decisions],
                                  new_todos=[], new_questions=[], updates=[], next_asks=[])

        monkeypatch.setattr(mo, "ClaudeCliClient", lambda *args, **kwargs: FakeClaude())

    def test_review_adds_what_the_rolling_summary_missed(self, ready, monkeypatch):
        orch, _ = ready
        self._fake_client(monkeypatch)
        result = orch.review("手動")
        assert result["ok"] is True
        assert result["decisions"] == 1
        assert orch._state_loop.state().decisions[0].text.startswith("台湾向け LP")

    def test_review_without_claude_cli_says_so(self, ready, monkeypatch):
        orch, _ = ready
        self._fake_client(monkeypatch, available=False)
        assert orch.review("手動") == {"ok": False, "reason": "Claude CLI が見つかりません"}

    def test_two_reviews_do_not_overlap(self, ready, monkeypatch):
        orch, _ = ready
        self._fake_client(monkeypatch)
        orch._review_lock.acquire()
        try:
            assert orch.review("手動")["ok"] is False
        finally:
            orch._review_lock.release()

    def test_planned_minutes_schedule_one_automatic_review(self, ready, monkeypatch):
        orch, _ = ready
        self._fake_client(monkeypatch)
        plan = orch.set_planned_minutes(60)
        assert plan["at"] == pytest.approx(orch._started_at + 50 * 60, abs=1)

        fired: list[str] = []
        monkeypatch.setattr(orch, "review", lambda reason="手動": fired.append(reason))
        orch._maybe_auto_review(orch._started_at + 49 * 60)   # まだ
        orch._maybe_auto_review(orch._started_at + 51 * 60)   # ここで 1 回
        orch._maybe_auto_review(orch._started_at + 55 * 60)   # 2 回目は走らない
        import time as _time
        _time.sleep(0.2)
        assert fired == ["終了 10 分前"]

    def test_clearing_the_plan_stops_the_automatic_review(self, ready, monkeypatch):
        orch, _ = ready
        orch.set_planned_minutes(60)
        assert orch.set_planned_minutes(None)["at"] is None
        fired: list[str] = []
        monkeypatch.setattr(orch, "review", lambda reason="手動": fired.append(reason))
        orch._maybe_auto_review(orch._started_at + 10 ** 6)
        assert fired == []


class TestFinishFromTheDashboard:
    """ターミナルで Ctrl+C を押さずに済ませる（運用者 依頼 2026-09-12）。"""

    def test_finish_sets_the_flag_the_loop_watches(self, orch):
        assert orch._finish_requested.is_set() is False
        assert orch.finish()["ok"] is True
        assert orch._finish_requested.is_set() is True


class TestAfterMeeting:
    def test_recording_is_rebuilt_by_default(self, orch, monkeypatch):
        """会議後は録音から全文を起こし直す（会議中 0.7 倍速・会議の外 12 倍速）。"""
        calls: list[list[str]] = []
        monkeypatch.setattr(mo, "wait_until_quiet", lambda names, **kwargs: {"reason": "落ち着いた", "waited_sec": 0.0})
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: calls.append([str(a) for a in argv]))
        orch._finalize_from_recording()
        assert calls and calls[0][1].endswith("finalize_meeting.py")
        assert calls[0][2] == str(orch.session_dir)

    def test_rebuild_can_be_turned_off(self, orch, monkeypatch):
        orch.meeting.finalize_after = False
        calls: list[list[str]] = []
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: calls.append(list(argv)))
        orch._show_outputs(orch.session_dir / "minutes_handoff.json")   # 終了時の表示だけ
        assert all("finalize_meeting.py" not in " ".join(map(str, c)) for c in calls)

    def test_outputs_are_listed_and_the_folder_opens(self, orch, monkeypatch, capsys, tmp_path):
        (tmp_path / "minutes_input.md").write_text("材料", encoding="utf-8")
        opened: list[list[str]] = []
        # ふだんのテストでは Finder を開かない（conftest で止めている）。ここだけ、開く経路を確かめる
        monkeypatch.delenv("MEETING_NO_OPEN", raising=False)
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: opened.append([str(a) for a in argv]))
        orch._show_outputs(tmp_path / "minutes_handoff.json")
        out = capsys.readouterr().out
        assert "議事録の材料" in out and "minutes_input.md" in out
        assert opened == [["open", str(orch.session_dir)]]

    def test_議事録へ渡すのは作り直しが終わってから(self, orch, monkeypatch, tmp_path):
        """前は議事録の机を先に開いていたので、minutes_input.md の書き換え中（約 100 秒）に
        議事録化が読み始めていた。順番を固定する。"""
        order: list[str] = []
        orch._bridge = None                              # テストから外部へ通知しない
        orch.meeting.finalize_after = True
        monkeypatch.setattr(orch, "_finalize_from_recording", lambda: order.append("作り直し"))
        monkeypatch.setattr(orch, "_hand_over_minutes", lambda handoff: order.append("議事録へ渡す"))
        monkeypatch.setattr(orch, "_show_outputs", lambda handoff: order.append("表示"))
        monkeypatch.setattr(orch, "_warn_failed_enrollments", lambda path: None)

        orch._shutdown(tmp_path / "speakers.json")

        assert order == ["作り直し", "議事録へ渡す", "表示"]

    def test_議事録まで作る設定では外へ出すかを会議の承認に合わせる(self, orch, monkeypatch):
        """「手元で作る」を選んだ会議で、会議のあとに黙って外へ出さない。"""
        calls: list[list[str]] = []
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: calls.append([str(a) for a in argv]))
        orch._bridge = None
        orch._dispatch_cfg.on_finish = "minutes"

        orch._llm_engine = "local"
        orch._hand_over_minutes(orch.session_dir / "minutes_handoff.json")
        orch._llm_engine = "gemini"
        orch._hand_over_minutes(orch.session_dir / "minutes_handoff.json")

        assert calls[0][1].endswith("make_minutes.py") and "--no-external" in calls[0]
        assert "--approved-external" in calls[1]

    def test_folder_opening_can_be_turned_off(self, orch, monkeypatch):
        orch.meeting.open_folder_after = False
        opened: list[list[str]] = []
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: opened.append(list(argv)))
        orch._show_outputs(orch.session_dir / "minutes_handoff.json")
        assert opened == []


class TestPreflightOnScreen:
    """起動セルフチェックを画面で見て画面で決める（運用者 依頼 2026-09-12）。"""

    @pytest.fixture
    def measured(self, orch, monkeypatch):
        def fake_measure():
            return {"mic": "[6] Headset One", "mic_name": "Headset One", "remote": "ScreenCaptureKit",
                    "mic_dbfs": -90.3, "remote_dbfs": -21.7,
                    "warnings": ["自分のマイク（Headset One）から音が来ていません。"]}

        monkeypatch.setattr(orch, "_measure_preflight", fake_measure)
        orch._ui_thread = object()   # 画面が動いている状態
        return orch

    def test_screen_choice_start_begins_the_meeting(self, measured):
        threading.Timer(0.05, lambda: measured.answer_preflight("start")).start()
        assert measured._preflight() is True

    def test_screen_choice_cancel_stops(self, measured):
        threading.Timer(0.05, lambda: measured.answer_preflight("cancel")).start()
        assert measured._preflight() is False

    def test_retry_measures_again(self, orch, monkeypatch):
        orch._ui_thread = object()
        results = [
            {"mic": "x", "remote": "y", "warnings": ["無音です"]},
            {"mic": "x", "remote": "y", "warnings": []},          # 話してから測り直したら問題なし
        ]
        monkeypatch.setattr(orch, "_measure_preflight", lambda: results.pop(0))
        threading.Timer(0.05, lambda: orch.answer_preflight("retry")).start()
        assert orch._preflight() is True
        assert results == []

    def test_unknown_choice_is_rejected(self, orch):
        with pytest.raises(ValueError):
            orch.answer_preflight("なんとなく")

    def test_without_the_dashboard_it_asks_in_the_terminal(self, orch, monkeypatch):
        """画面が無いときは従来どおり端末で聞く。"""
        orch._ui_thread = None
        monkeypatch.setattr(orch, "_measure_preflight", lambda: {"mic": "x", "remote": "y", "warnings": ["無音です"]})
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        assert orch._preflight() is True

    def test_no_warnings_starts_without_asking(self, orch, monkeypatch):
        orch._ui_thread = object()
        monkeypatch.setattr(orch, "_measure_preflight", lambda: {"mic": "x", "remote": "y", "warnings": []})
        assert orch._preflight() is True


class TestLateStart:
    """会議が始まってからあわてて起動する場合（運用者 2026-09-12）。"""

    def test_start_now_skips_the_measurements(self, orch, monkeypatch):
        orch._ui_thread = object()
        monkeypatch.setattr(orch, "_check_output_routing", lambda: [])
        orch.mic.matched_candidate = "Headset One"
        threading.Timer(0.05, lambda: orch.answer_preflight("start_now")).start()
        assert orch._preflight() is True   # 測定なしで本編へ

    def test_silence_after_start_is_warned_once(self, orch):
        """開始後に音が来ないソースを見張る（測定を飛ばしたときの唯一の砦）。"""
        orch.meeting.silence_alert_sec = 60.0
        events: list[dict] = []
        orch.bus.publish = lambda kind, data: events.append({"kind": kind, **data})
        silent = {SELF_KEY: [np.zeros(4800, dtype=np.float32)], REMOTE_KEY: [_voice(0.1)]}

        orch._watch_silence(orch._started_at + 30, silent)     # まだ
        assert events == []
        orch._watch_silence(orch._started_at + 90, silent)     # 1 回だけ警告
        orch._watch_silence(orch._started_at + 150, silent)
        warnings = [e for e in events if e.get("warnings")]
        assert len(warnings) == 1
        assert "自分のマイク" in warnings[0]["warnings"][0]

    def test_recovery_is_announced(self, orch):
        orch.meeting.silence_alert_sec = 60.0
        events: list[dict] = []
        orch.bus.publish = lambda kind, data: events.append({"kind": kind, **data})
        orch._watch_silence(orch._started_at + 90, {SELF_KEY: [np.zeros(4800, dtype=np.float32)]})
        orch._watch_silence(orch._started_at + 100, {SELF_KEY: [_voice(0.1)]})
        assert any("音が戻りました" in (e.get("note") or "") for e in events)


class TestWaitForOtherApps:
    """MacWhisper の文字起こしが落ち着くまで待ってから作り直す（運用者 判断 2026-09-12）。"""

    def test_default_is_not_to_wait(self, orch):
        """2026-09-15: MacWhisper を常駐させない方針になったので、既定では誰も待たない。"""
        assert orch.meeting.wait_for_apps == []

    def test_rebuild_waits_then_runs(self, orch, monkeypatch):
        orch.meeting.wait_for_apps = ["MacWhisper"]   # 常駐させる人の設定
        order: list[str] = []
        monkeypatch.setattr(mo, "wait_until_quiet", lambda names, **kwargs: order.append(f"待つ:{names}") or {"reason": "落ち着いた", "waited_sec": 120.0})
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: order.append("作り直す"))
        orch._finalize_from_recording()
        assert order == ["待つ:['MacWhisper']", "作り直す"]

    def test_waiting_can_be_turned_off(self, orch, monkeypatch):
        orch.meeting.wait_for_apps = []
        called: list[str] = []
        monkeypatch.setattr(mo, "wait_until_quiet", lambda *a, **k: called.append("待った") or {"reason": "落ち着いた", "waited_sec": 0.0})
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: None)
        orch._finalize_from_recording()
        assert called == []

    def test_timeout_still_proceeds(self, orch, monkeypatch, capsys):
        """待ち続けて何も出ないより、上限で先へ進む。"""
        orch.meeting.wait_for_apps = ["MacWhisper"]
        monkeypatch.setattr(mo, "wait_until_quiet", lambda names, **kwargs: {"reason": "待ち時間の上限", "waited_sec": 1800.0})
        ran: list[str] = []
        monkeypatch.setattr(mo.subprocess, "run", lambda argv, **kwargs: ran.append("作り直す"))
        orch._finalize_from_recording()
        assert ran == ["作り直す"]
        assert "待ち時間の上限" in capsys.readouterr().out


class TestLiveBatchStt:
    """会議中の文字起こしを外（Gemini）で回す経路（方式④＝30 秒刻み）。

    見るのは 3 つ。①外へ送るのは音声だけで**話者は手元で当てる** ②返ってきた時刻が
    会議の時計に戻る ③送れなかった区間が記録されて、あとで拾い直せる。
    """

    def _window(self, amplitude: float = 0.5):
        from src.stt.live_batch import AudioWindow, Span

        # 0〜2 秒と 10〜12 秒に発話（無音は送らないので、繋いだ音声は 4 秒）
        audio = np.full(int(4 * SR), amplitude, dtype=np.float32)
        return AudioWindow(key=REMOTE_KEY, audio=audio, sample_rate=SR,
                           spans=[Span(0.0, 0.0, 2.0), Span(2.0, 10.0, 2.0)])

    def test_話者は手元で当てて画面と全文へ流す(self, orch, tmp_path):
        # ここへ来る行は既に会議の時計（無音を跨いだ 10 秒台がある）
        rows = [{"speaker": "", "text": "そこは来週までに", "start_time": 0.5, "end_time": 1.5},
                {"speaker": "", "text": "お願いします", "start_time": 10.5, "end_time": 11.5}]

        orch._on_live_rows(REMOTE_KEY, self._window(), rows)

        # 外が返した話者（空）ではなく、手元の声紋の名前が入る
        assert [segment.speaker for segment in orch._transcript_buffer] == ["田中", "田中"]
        assert [segment.start_time for segment in orch._transcript_buffer] == [0.5, 10.5]
        lines = (tmp_path / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and "田中" in lines[0]

    def test_自分側は判定せず固定ラベルのまま(self, orch):
        window = self._window(amplitude=0.9)   # 声紋では「不明話者1」になる振幅
        orch._on_live_rows(SELF_KEY, window, [{"speaker": "話者1", "text": "はい", "start_time": 0.5, "end_time": 1.0}])

        assert orch._transcript_buffer[0].speaker == "自分"
        assert orch.diarizer.calls == 0     # 自分側はマイクが別なので声紋にかけない

    def test_直したときのために行の声を覚える(self, orch):
        """画面で名前を付けたとき、その声を過去の行へ反映できるようにする。"""
        orch._on_live_rows(REMOTE_KEY, self._window(), [
            {"speaker": "", "text": "確認します", "start_time": 0.5, "end_time": 1.5}])

        assert orch._row_voice(0.5) is not None

    def test_送れなかった区間は記録されて画面にも出る(self, orch, tmp_path):
        events = orch.bus.subscribe()
        orch._on_live_gap(REMOTE_KEY, self._window(), RuntimeError("HTTP 503"))

        entry = json.loads((tmp_path / mo.GAPS_FILE).read_text(encoding="utf-8").splitlines()[0])
        assert entry["stream"] == REMOTE_KEY
        assert (entry["start_time"], entry["end_time"]) == (0.0, 12.0)
        assert "503" in entry["error"]
        assert "送れなかった" in events.get(0.1).data["warnings"][0]

    def test_外で起こすときは手元のモデルを呼ばない(self, orch):
        """チャンクは溜めるだけ。GPU を使わないのが方式④の要点。"""
        fed: list = []
        orch._live_stt = type("Fake", (), {"feed": lambda self, key, chunk: fed.append((key, chunk))})()
        called = []
        orch._executor.submit = lambda *args, **kwargs: called.append(args)

        orch._submit(REMOTE_KEY, AudioChunk(REMOTE_KEY, np.ones(SR, dtype=np.float32), 0.0, 1.0, SR))

        assert len(fed) == 1 and not called

    def test_承認が取れなければ手元へ落ちる(self, orch, monkeypatch):
        """関門を通らなければ送らない。会議は手元の Whisper と Ollama で続ける（失われてはいけない）。"""
        loaded = []
        monkeypatch.setattr(orch.whisper, "_ensure_model", lambda: loaded.append(True))
        monkeypatch.setattr(orch.llm, "health_check", lambda: True)
        orch._stt_engine = orch._llm_engine = "gemini"
        orch._external_cfg.enabled = False      # 設定で切られている＝聞くまでもなく送らない

        orch._start_external_engines()

        assert orch._live_stt is None and orch._stt_engine == "local" and loaded
        assert orch._llm_engine == "local"      # 要約も手元のまま
        assert isinstance(orch.llm, mo.OllamaClient)

    def test_外で回すものは一度の確認にまとまる(self, orch, monkeypatch):
        """音声と全文の両方を出すときも、聞くのは 1 回（何を出すかは文面に並べる）。"""
        asked: list[list[str]] = []
        monkeypatch.setattr(orch.whisper, "_ensure_model", lambda: None)
        monkeypatch.setattr(orch.llm, "health_check", lambda: True)
        monkeypatch.setattr(orch, "_ask_external_live",
                            lambda config, items=None: asked.append(list(items or [])) or False)
        orch._stt_engine = orch._llm_engine = "gemini"
        orch._verify_engine = "none"            # 🔍 の手段は別のテストで見る（この機械に claude があるかで変わる）
        orch._external_cfg.enabled = True
        orch._external_cfg.billing_project = "your-gcp-project"

        orch._start_external_engines()

        assert len(asked) == 1
        assert asked[0] == ["会議中の音声（30 秒ごと）",
                            "会議中の全文と事前資料（要約のため。音声は送りません）"]

    def test_議事録まで外で作るなら確認の文面にそう書く(self, orch, monkeypatch):
        """会議のあとの議事録も同じ承認で Gemini に回る。文面に無いものは送らない。"""
        asked: list[list[str]] = []
        monkeypatch.setattr(orch.llm, "health_check", lambda: True)
        monkeypatch.setattr(orch, "_ask_external_live",
                            lambda config, items=None: asked.append(list(items or [])) or False)
        orch._llm_engine = "gemini"
        orch._verify_engine = "none"
        orch._dispatch_cfg.on_finish = "minutes"
        orch._external_cfg.enabled = True
        orch._external_cfg.billing_project = "your-gcp-project"

        orch._start_external_engines()

        assert asked == [["会議中の全文と事前資料（要約と会議のあとの議事録のため。音声は送りません）"]]


def test_起動前の警告はセルフチェックの画面に混ざる(orch, monkeypatch):
    """古い事前資料（4 月の別会議の計画）に、開始を選ぶ前に気づけるようにする。"""
    orch.startup_warnings = ["事前資料が今日のものではありません: 2026-04-11-toria.md（153 日前）"]
    monkeypatch.setattr(orch, "_wait_preflight_action", lambda phase, payload: "skip")
    orch.mic = type("D", (), {"index": 6, "name": "Headset One", "channels": 1,
                              "matched_candidate": "Headset One", "__str__": lambda self: "mic"})()

    result = orch._measure_preflight()

    assert any("2026-04-11-toria.md" in warning for warning in result["warnings"])


def _fail_llm(*args, **kwargs):
    """要約のエンジンが使えない状況（Ollama が居ない・キーが無い）を作る。"""
    raise RuntimeError("LLM は使えません")


class TestPrepFromTheDashboard:
    """事前資料を画面から足す（会議ごとにリセットし、途中でも足せる）。

    2026-09-11 の事故の裏返し: 共有フォルダの資料を使い回して、4 月の別会議の
    インタビュー計画が定例の評価基準になっていた。
    """

    def test_足した資料の固有名詞がこの会議の守り札と要約の手がかりになる(self, orch, tmp_path):
        """09-14 の会議で社名「MDクラウド」が辞書の「クラウド→Claude」で「MDClaude」になった。"""
        orch._glossary = [("クラウド", "Claude")]
        orch.attach_prep("議題.md", "## 用語\n- MDクラウド\n\n## 議題\n- Revuno の有料プラン\n")

        orch._emit_segments([TranscriptSegment("参加者C", "MDクラウドの方で、クラウドに上げます", 1.0, 2.0, "")])

        assert orch._transcript_buffer[-1].text == "MDクラウドの方で、Claudeに上げます"
        assert "この会議の固有名詞" in orch._prep_for_prompt and "Revuno" in orch._prep_for_prompt
        assert json.loads((tmp_path / "prep_terms.json").read_text(encoding="utf-8"))[:2] == ["MDクラウド", "Revuno"]

    def test_画面で付けた名前も守り札になる(self, orch):
        orch._glossary = [("クラウド", "Claude")]
        orch._aliases["不明話者1"] = "エムディクラウド 参加者C"

        assert "エムディクラウド" in orch._protected_words()

    def test_足した資料がその場で要約の材料になる(self, orch):
        orch.attach_prep("議題.md", "## 議題\n- 見積の確認\n")

        assert "見積の確認" in orch._prior_tasks
        assert [file["name"] for file in orch.prep_files()] == ["議題.md"]

    def test_会議の途中でも動いている要約へ渡る(self, orch):
        """次の窓のプロンプトから効く（止めずに差し替えられる）。"""
        class FakeLoop:
            def __init__(self):
                self.updater = type("U", (), {"prior_tasks": "（前回タスク・アジェンダ未設定）"})()

        orch._state_loop = FakeLoop()
        orch.attach_prep("追加メモ.txt", "先方の予算は未確定")

        assert "先方の予算は未確定" in orch._state_loop.updater.prior_tasks

    def test_外すと材料からも消える(self, orch):
        orch.attach_prep("議題.md", "見積の確認")
        orch.remove_prep("議題.md")

        assert orch.prep_files() == []
        assert "見積の確認" not in orch._prior_tasks

    def test_渡っている文字数を返す(self, orch, monkeypatch):
        """全文が渡るとは限らない。黙って切られると「付けたのに効かない」の理由が分からない。"""
        monkeypatch.setattr(orch.llm, "chat_json", _fail_llm)     # まとめは使えない状況にする
        orch._state_cfg.prior_tasks_chars = 50
        status = orch.attach_prep("長い資料.md", "あ" * 400)

        assert status["chars"] > status["used_chars"] == 50
        assert status["digested"] is False

    def test_長い資料は足したときに一度だけまとめる(self, orch, monkeypatch):
        """実物の提案書 PDF は 10 万字。毎窓に載るのは 2,000 字なので、そのままでは 2% しか見ない。

        全文を毎窓に載せると 30 秒ごとに課金され、手元の Ollama では文脈にも入らない。
        ∴ 足したときに 1 回だけまとめ、以後はまとめだけを載せる。
        """
        calls = []

        def fake_chat_json(system, user, schema, **kwargs):
            calls.append((system, user))
            return {"purpose": "次期サイトの提案", "points": ["公開時期"], "numbers": ["300 万円"],
                    "tasks": ["見積を 9/20 までに"], "terms": ["自社"]}

        monkeypatch.setattr(orch.llm, "chat_json", fake_chat_json)
        orch._state_cfg.prior_tasks_chars = 200

        status = orch.attach_prep("提案書.md", "あ" * 5000)

        assert len(calls) == 1                       # 1 回だけ（毎窓ではない）
        assert "見積を 9/20 までに" in orch._prep_for_prompt
        assert "300 万円" in orch._prep_for_prompt    # 数字は原文のまま残す
        assert len(orch._prep_for_prompt) < 500
        assert status["digested"] is True
        assert "まとめました" in status["note"]

    def test_外で要約を回し始めたら長い資料をまとめ直す(self, orch, monkeypatch):
        """承認前に足した資料は手元でまとめられる（Ollama が止まっていれば先頭 2,000 字だけ）。承認後に外でまとめ直す。"""
        monkeypatch.setattr(orch.llm, "chat_json", _fail_llm)          # 会議の前: 手元ではまとめられない
        orch._state_cfg.prior_tasks_chars = 200
        before = orch.attach_prep("みなととのやりとり.md", "あ" * 5000)
        assert before["digested"] is False

        class FakeGemini:
            config = type("C", (), {"model": "gemini-3.8-flash"})()
            engines = []

            def chat_json(self, system, user, schema, **kwargs):
                FakeGemini.engines.append(len(user))
                return {"purpose": "みなと定例", "points": [], "numbers": [], "tasks": ["前回の宿題"], "terms": []}

        monkeypatch.setattr(mo, "GeminiClient", FakeGemini)
        orch.llm = FakeGemini()

        orch._redigest_prep(wait=True)

        assert FakeGemini.engines == [len(orch._prep.text)]            # 外なので全文を読ませる
        assert "前回の宿題" in orch._prep_for_prompt

    def test_短い資料はまとめ直さない(self, orch, monkeypatch):
        orch.attach_prep("議題.md", "見積の確認")
        started = []
        monkeypatch.setattr(mo.threading, "Thread", lambda *args, **kwargs: started.append(kwargs) or None)

        orch._redigest_prep()

        assert started == []

    def test_まとめに失敗しても会議は続く(self, orch, monkeypatch):
        monkeypatch.setattr(orch.llm, "chat_json", _fail_llm)
        orch._state_cfg.prior_tasks_chars = 100

        status = orch.attach_prep("提案書.md", "あ" * 5000)

        assert orch._prep_for_prompt.startswith("### 提案書")   # 全文のまま（先頭だけ使われる）
        assert "失敗" in status["note"]

    def test_手元でまとめるときは読ませる量を切る(self, orch, monkeypatch):
        """手元のモデルは文脈が狭い。渡しすぎると黙って頭から捨てられる＝読んだつもりで読んでいない。"""
        seen = []
        monkeypatch.setattr(orch.llm, "chat_json", lambda system, user, schema, **kwargs: (
            seen.append(len(user)) or {"purpose": "", "points": [], "numbers": [], "tasks": ["宿題"], "terms": []}))
        orch._llm_engine = "local"
        status = orch.attach_prep("大きい資料.md", "あ" * 120_000)

        assert seen == [mo.PREP_DIGEST_LOCAL_INPUT]
        assert "先頭 30,000 文字まで" in status["note"]

    def test_まとめは動いている要約へも渡る(self, orch, monkeypatch):
        class FakeLoop:
            def __init__(self):
                self.updater = type("U", (), {"prior_tasks": ""})()

        monkeypatch.setattr(orch.llm, "chat_json", lambda *args, **kwargs: {
            "purpose": "定例", "points": [], "numbers": [], "tasks": ["宿題 A"], "terms": []})
        orch._state_loop = FakeLoop()
        orch._state_cfg.prior_tasks_chars = 100
        orch.attach_prep("資料.md", "い" * 3000)

        assert "宿題 A" in orch._state_loop.updater.prior_tasks

    def test_読めない種類は断る(self, orch):
        with pytest.raises(ValueError):
            orch.attach_prep("写真.heic", "…")

    def test_資料はファイルとして受け取って本文を取り出す(self, orch, tmp_path):
        """PDF・Word・Excel・PowerPoint は手元で本文を取り出す（外へは出さない）。"""
        from tests.test_documents import _zip

        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        docx = _zip({"word/document.xml":
                     f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>'
                     f'<w:p><w:r><w:t>見積は 9/20 まで</w:t></w:r></w:p></w:body></w:document>'})

        status = orch.attach_prep_file("見積メモ.docx", docx)

        assert "見積は 9/20 まで" in orch._prior_tasks
        assert [file["name"] for file in status["files"]] == ["見積メモ.md"]
        # 元のファイルも残す（何を材料にしたかを後から辿るため）
        assert (tmp_path / "prep" / "originals" / "見積メモ.docx").exists()

    def test_文字の入っていない資料は断る(self, orch):
        """スキャンしただけの PDF。黙って空を読み込むと「付けたのに効かない」になる。"""
        from tests.test_documents import _pdf

        with pytest.raises(ValueError, match="スキャン"):
            orch.attach_prep_file("スキャン.pdf", _pdf([]))

    def test_大きすぎるファイルは断る(self, orch):
        with pytest.raises(ValueError, match="大きすぎ"):
            orch.attach_prep_file("巨大.pdf", b"0" * (mo.PREP_MAX_BYTES + 1))

    def test_空は断る(self, orch):
        with pytest.raises(ValueError):
            orch.attach_prep("空.md", "   ")

    def test_セッションの外へは書かせない(self, orch, tmp_path):
        """名前は画面から来る。パスを混ぜられてもセッションの中に収める。"""
        orch.attach_prep("../../逃げた.md", "中身")

        assert (tmp_path / "prep" / "逃げた.md").exists()
        assert not (tmp_path.parent / "逃げた.md").exists()

    def test_無いものを外そうとしたら教える(self, orch):
        with pytest.raises(KeyError):
            orch.remove_prep("無い.md")


class TestVerifyOnDemand:
    """画面で指した 1 行を Web で裏取りする（Claude CLI・オンデマンド）。

    自動では走らせない。2026-09-12 の実測で、自動検出は 50 分に 71 件拾って
    その大半が社内の見積・案件の話だった（外では確かめようがない）。
    """

    def _ready(self, orch, result=None, error=None):
        """調べる口（Claude CLI）だけを差し替える。裏取りの段取り自体は本物を通す。"""
        class FakeClient:
            def __init__(self):
                self.calls = []

            def available(self):
                return True

            def verify(self, quote, context="", **kwargs):
                self.calls.append((quote, context))
                if error is not None:
                    raise error
                return result or {"verdict": "要確認", "note": "2026-12-31 までの価格",
                                  "sources": ["https://ai.google.dev/gemini-api/docs/pricing"]}

        orch._verifier.client = FakeClient()
        return orch._verifier.client

    def _wait(self, events, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            event = events.get(0.2)
            if event is not None and event.type == "verify" and event.data.get("phase") == "done":
                return event.data
        raise AssertionError("裏取りの結果が来ませんでした")

    def test_指した行を調べて出典つきで返す(self, orch, tmp_path):
        verifier = self._ready(orch)
        orch._emit_segments([TranscriptSegment("田中", "入力は 100 万トークン 0.75 ドルらしい", 12.0, 15.0, "")])
        events = orch.bus.subscribe()

        accepted = orch.verify_row(12.0)
        assert accepted["ok"] is True and accepted["phase"] == "running"

        done = self._wait(events)
        assert done["verdict"] == "要確認"
        assert done["sources"] == ["https://ai.google.dev/gemini-api/docs/pricing"]
        assert verifier.calls[0][0] == "入力は 100 万トークン 0.75 ドルらしい"
        # 記録が残る（議事録に出典を添える材料）
        line = json.loads((tmp_path / mo.VERIFY_FILE).read_text(encoding="utf-8").splitlines()[0])
        assert line["verdict"] == "要確認" and line["quote"].startswith("入力は")

    def test_前後の発言も渡す(self, orch):
        """発言だけだと何の話か分からない。文脈として前後を添える（音声は送らない）。"""
        verifier = self._ready(orch)
        orch._emit_segments([
            TranscriptSegment("自分", "Gemini の価格の話ですが", 10.0, 11.0, ""),
            TranscriptSegment("田中", "入力は 0.75 ドルらしい", 12.0, 15.0, ""),
        ])
        events = orch.bus.subscribe()

        orch.verify_row(12.0)
        self._wait(events)

        assert "Gemini の価格の話ですが" in verifier.calls[0][1]

    def test_調べられなくても会議は止まらない(self, orch):
        self._ready(orch, error=RuntimeError("Claude CLI が失敗しました"))
        orch._emit_segments([TranscriptSegment("田中", "何かの主張", 20.0, 21.0, "")])
        events = orch.bus.subscribe()

        orch.verify_row(20.0)

        done = self._wait(events)
        assert done["verdict"] == "不明" and "調べられませんでした" in done["note"]

    def test_同じ行を二重に投げない(self, orch):
        self._ready(orch)
        orch._emit_segments([TranscriptSegment("田中", "主張", 30.0, 31.0, "")])
        with orch._verifier._lock:
            orch._verifier._running.add(30.0)

        assert orch.verify_row(30.0)["already"] is True

    def test_Claude_CLI_が無ければ断る(self, orch):
        orch._verifier.client = type("Missing", (), {"available": lambda self: False})()
        orch._emit_segments([TranscriptSegment("田中", "主張", 40.0, 41.0, "")])

        assert orch.verify_row(40.0)["ok"] is False

    def test_無い行は教える(self, orch):
        self._ready(orch)
        with pytest.raises(KeyError):
            orch.verify_row(999.0)


class TestVerifyEngine:
    """🔍 の手段を選ぶ（Claude CLI → Gemini＋Google 検索 → なし）。

    Claude のサブスクが無い人にも逃げ道を残す（2026-09-14 運用者 依頼）。ただし Gemini は
    **その会議で外へのテキスト送信を承認したとき**だけ。承認なしに発言を外へ出さない。
    """

    class FakeGemini:
        def __init__(self, text='{"verdict": "一致", "note": "公式の価格表どおり", "sources": ["https://記憶で書いた.example/404"]}'):
            self.text = text
            self.prompts: list[str] = []
            self.closed = False
            self.config = type("C", (), {"model": "gemini-3.8-flash"})()

        def search_json(self, prompt, thinking_budget=None):
            self.prompts.append(prompt)
            self.budget = thinking_budget
            return self.text, [{"uri": "https://ai.google.dev/gemini-api/docs/pricing", "title": "ai.google.dev"}]

        def close(self):
            self.closed = True

    def _without_claude(self, orch, monkeypatch, *, answer=True):
        """Claude CLI が無い機械で、外への送信を有効にした状態を作る。"""
        asked: list[list[str]] = []
        orch._verify_engine = "auto"
        orch._verifier.disable("Claude CLI が見つかりません")
        orch._external_cfg.enabled = True
        orch._external_cfg.billing_project = "your-gcp-project"
        fake = self.FakeGemini()
        monkeypatch.setattr(mo, "GeminiClient", lambda config: fake)

        def fake_approve(config, session_dir, files, ask=None):
            if not ask("聞く"):
                raise mo.ExternalSendRefused("断られました")
            from src.stt.external_consent import Approval
            return Approval(billing_project=config.billing_project, at="2026-09-14T18:00:00+09:00")

        monkeypatch.setattr(mo, "approve", fake_approve)
        monkeypatch.setattr(orch, "_ask_external_live",
                            lambda config, items=None: asked.append(list(items or [])) or answer)
        return fake, asked

    def test_サブスクが無くても承認すれば_Gemini_で調べる(self, orch, monkeypatch, tmp_path):
        fake, asked = self._without_claude(orch, monkeypatch)

        orch._start_external_engines()

        # 何を出すかは確認の文面に並ぶ（音声も要約も手元のままでも、🔍 のために聞く）
        assert asked == [["🔍 を押した行と前後の文字（Google 検索で裏取りするため）"]]
        assert isinstance(orch._verifier.client, mo.GeminiVerifyClient)
        assert orch.session_info()["models"]["verify"]["name"] == "Gemini＋Google 検索"
        sends = [json.loads(line) for line in (tmp_path / "external_sends.jsonl").read_text(encoding="utf-8").splitlines()]
        assert sends[-1]["note"] == "verify"

    def test_出典は検索で実際に使われたページだけ(self, orch, monkeypatch):
        """モデルが本文に書いた URL は記憶から出ることがある（実測で 29 本中 7 本が 404）。"""
        fake, _ = self._without_claude(orch, monkeypatch)
        orch._start_external_engines()
        orch._emit_segments([TranscriptSegment("田中", "入力は 100 万トークン 0.75 ドル", 12.0, 15.0, "")])
        events = orch.bus.subscribe()

        assert orch.verify_row(12.0)["engine"] == "Gemini＋Google 検索"
        done = TestVerifyOnDemand()._wait(events)

        assert done["verdict"] == "一致"
        assert done["sources"] == ["https://ai.google.dev/gemini-api/docs/pricing"]
        assert "入力は 100 万トークン 0.75 ドル" in fake.prompts[0]

    def test_断ったら_Gemini_には出さず理由を返す(self, orch, monkeypatch):
        fake, _ = self._without_claude(orch, monkeypatch, answer=False)

        orch._start_external_engines()
        orch._emit_segments([TranscriptSegment("田中", "主張", 40.0, 41.0, "")])

        refused = orch.verify_row(40.0)
        assert refused["ok"] is False and "承認" in refused["reason"]
        assert not fake.prompts
        assert orch.session_info()["models"]["verify"]["available"] is False

    def test_Claude_CLI_があれば_auto_では聞かない(self, orch, monkeypatch):
        """サブスクで足りるものを、わざわざ外（従量）へ出さない。"""
        _, asked = self._without_claude(orch, monkeypatch)
        claude = type("Claude", (), {"available": lambda self: True, "label": "Claude CLI"})()
        orch._verifier.use(claude)

        orch._start_external_engines()

        assert asked == [] and orch._verifier.client is claude

    def test_none_なら使わないことを画面に出す(self, orch, monkeypatch):
        _, asked = self._without_claude(orch, monkeypatch)
        orch._verify_engine = "none"
        orch._verifier.disable(orch._verify_unavailable_reason(True))

        orch._start_external_engines()
        orch._emit_segments([TranscriptSegment("田中", "主張", 50.0, 51.0, "")])

        assert asked == []
        assert "verify_engine: none" in orch.verify_row(50.0)["reason"]

    def test_検索しなかったら一致とは言わない(self):
        """思考 0 の Gemini は google_search を付けても検索せず記憶で答えた（2026-09-15 実測・出典 0 件）。"""
        fake = self.FakeGemini(text='{"verdict": "一致", "note": "価格表どおり", "sources": []}')
        fake.search_json = lambda prompt, thinking_budget=None: (fake.text, [])

        result = mo.GeminiVerifyClient(fake).verify("主張")

        assert result["verdict"] == "要確認" and result["note"].startswith("（検索されず")

    def test_裏取りだけは思考をおまかせにして検索させる(self):
        fake = self.FakeGemini()

        mo.GeminiVerifyClient(fake).verify("主張")

        assert fake.budget == -1 and "必ず Google 検索" in fake.prompts[0]

    def test_知らない判定は通さない(self):
        client = mo.GeminiVerifyClient(self.FakeGemini(text='{"verdict": "たぶん正しい", "note": "", "sources": []}'))

        with pytest.raises(ValueError):
            client.verify("主張")


class TestVoiceLibraryInTheMeeting:
    """前の会議で覚えた人を、不明話者に「候補として」出す。自動では名前を付けない。"""

    @pytest.fixture
    def with_library(self, orch, tmp_path):
        from src.audio.voice_library import VoiceLibrary, VoiceLibraryConfig

        person_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        library = VoiceLibrary(tmp_path / "voices.json", VoiceLibraryConfig(enabled=True))
        library.learn("2026-09-14_1359", {"参加者C": [person_c + 0.01 * i for i in range(6)]})
        orch._voice_cfg = library.config
        orch._voices = library
        orch.diarizer.unknown_names = ["不明話者1"]
        orch.diarizer.speaker_names = ["田中", "佐藤", "不明話者1"]
        orch.diarizer.voice_of = lambda name: (np.array([0.97, 0.2, 0.0], dtype=np.float32), 3) if name == "不明話者1" else None
        return library

    def test_不明話者に候補が付く(self, orch, with_library):
        orch._speaker_stats["不明話者1"] = [3, 40]

        item = next(item for item in orch.participants() if item["name"] == "不明話者1")

        assert [s["name"] for s in item["suggestions"]] == ["参加者C"]
        assert item["suggestions"][0]["last_session"] == "2026-09-14_1359"

    def test_声が育つまでは候補を出さない(self, orch, with_library):
        orch.diarizer.voice_of = lambda name: (np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)

        assert orch.voice_suggestions("不明話者1") == []

    def test_押したら名前が付き記録が残る(self, orch, with_library, tmp_path, monkeypatch):
        renamed = []
        monkeypatch.setattr(orch, "rename_speaker", lambda old, new: renamed.append((old, new)))

        orch.accept_voice("不明話者1", "参加者C")

        assert renamed == [("不明話者1", "参加者C")]
        entry = json.loads((tmp_path / "voice_matches.jsonl").read_text(encoding="utf-8"))
        assert entry["accepted"] is True and entry["score"] > 0.9

    def test_違うと押した候補はもう出さない(self, orch, with_library, tmp_path):
        orch.reject_voice("不明話者1", "参加者C")

        assert orch.voice_suggestions("不明話者1") == []
        assert json.loads((tmp_path / "voice_matches.jsonl").read_text(encoding="utf-8"))["accepted"] is False

    def test_台帳が無効なら何もしない(self, orch):
        assert orch._voices is None and orch.voice_suggestions("不明話者1") == []

    def test_会議のあとに改名後の名前で覚える(self, orch, with_library, tmp_path, monkeypatch):
        """作り直していない全文は改名前の名前のまま。会議中の改名を当ててから覚える。"""
        import wave

        with wave.open(str(tmp_path / "recording_remote.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(np.zeros(16000 * 60, dtype=np.int16).tobytes())
        rows = [{"speaker": "不明話者2", "text": "x", "start_time": float(i * 5), "end_time": float(i * 5 + 4)}
                for i in range(8)]
        (tmp_path / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        orch._aliases["不明話者2"] = "参加者D"
        monkeypatch.setattr(mo.EnrolledDiarizer, "embed", staticmethod(lambda clip, sr: np.array([0.0, 1.0, 0.0], dtype=np.float32)))

        orch._learn_voices()

        assert "参加者D" in with_library.people and with_library.people["参加者D"][0]["rows"] == 8


class TestFinishGate:
    """会議の終わりは、押してから仕上げる（勝手に数分の処理を始めない）。"""

    def _meeting_files(self, tmp_path):
        (tmp_path / "transcripts.jsonl").write_text("{}", encoding="utf-8")
        (tmp_path / "minutes_input.md").write_text("# 材料", encoding="utf-8")

    def test_返事が無ければ後回しにする(self, orch, tmp_path, monkeypatch):
        """勝手に始めない。移動する・Wi-Fi が切れる場面で途中で落ちるのを避ける。"""
        self._meeting_files(tmp_path)
        orch._ui_thread = object()
        orch.meeting.finish_wait_sec = 0.01
        monkeypatch.setattr(orch._finish_decided, "wait", lambda timeout: False)

        assert orch._ask_finish() is False

    def test_押されたら仕上げる(self, orch, tmp_path, monkeypatch):
        self._meeting_files(tmp_path)
        orch._ui_thread = object()
        monkeypatch.setattr(orch._finish_decided, "wait", lambda timeout: orch.finish_now("now"))

        assert orch._ask_finish() is True

    def test_画面が無ければ従来どおり走らせる(self, orch, tmp_path):
        """端末で見ている＝待てる状況。ここで止めると無人の流し込みが進まない。"""
        self._meeting_files(tmp_path)
        orch._ui_thread = None

        assert orch._ask_finish() is True

    def test_設定で常に走らせることもできる(self, orch, tmp_path):
        self._meeting_files(tmp_path)
        orch._ui_thread = object()
        orch.meeting.finish_mode = "auto"

        assert orch._ask_finish() is True

    def test_残りが無ければ聞かずに進む(self, orch, tmp_path):
        """もう終わっている会議（作り直しも議事録も済み）では、押させない。"""
        self._meeting_files(tmp_path)
        for name in ("transcripts_final.jsonl", "glossary_candidates.json", "minutes.md"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        orch._ui_thread = object()
        orch.meeting.finalize_after = False

        # 声の台帳だけはファイルで終わりを判定できないので、使う設定ならその1件だけ残る
        assert [step.key for step in mo.finish.remaining(tmp_path)] == ["voices"]
        # 使わない設定（配布版の既定）なら、残らない
        assert mo.finish.remaining(tmp_path, voice_enabled=False) == []

    def test_声の台帳を使わないなら声の工程は残さない(self, orch, tmp_path):
        """2026-09-20 の配布版の受け入れ確認で発見。

        `voice_library.enabled: false`（配布版の既定）だと台帳に誰も入らないので、
        「話した人の声を台帳に覚える」が**永久に残り**、毎回「仕上げが残っています」と出ていた。
        """
        self._meeting_files(tmp_path)
        orch._voice_cfg.enabled = False

        orch._stop_here(tmp_path / "minutes_handoff.json")

        keys = [step["key"] for step in mo.finish.read_pending(tmp_path)["steps"]]
        assert "voices" not in keys

    def test_後回しにしたら残りと再開の仕方を残す(self, orch, tmp_path, capsys):
        self._meeting_files(tmp_path)

        orch._stop_here(tmp_path / "minutes_handoff.json")

        pending = mo.finish.read_pending(tmp_path)
        # 声の台帳は既定 off なので「声を覚える」は並ばない（下のテストを見よ）
        assert [step["key"] for step in pending["steps"]] == ["finalize", "glossary", "minutes"]
        out = capsys.readouterr().out
        assert "録音と全文は残っています" in out and "会議アシスタント.command" in out


class TestLiveSttLabel:
    """画面の「モデル」欄は、**いま音声がどこへ行っているか**を出す。

    2026-09-20 に発見: Deepgram で回していても「外（Gemini・30 秒刻み）」と出ていた
      （09-18 に Deepgram とエンジンごとの刻みを入れたとき、表示側を直し忘れた）。
      この道具は「どこへ出るか」を売りにしているので、ここの取り違えは重い。
    """

    def _start(self, orch, monkeypatch, provider: str) -> dict:
        monkeypatch.setattr(mo, "load_key", lambda path: "dummy-key")
        monkeypatch.setattr(mo, "record_send", lambda *a, **k: None)
        monkeypatch.setattr(mo, "TranscribeApi", lambda key, config: object())
        monkeypatch.setattr(mo, "LiveBatchStt", lambda *a, **k: object())
        if provider == "deepgram":
            import src.stt.deepgram_transcribe as dg
            monkeypatch.setattr(dg, "DeepgramApi", lambda key, config: object())
        orch._external_cfg.provider = provider
        orch._external_cfg.deepgram_key_file = "dummy.key"
        orch._external_cfg.key_file = "dummy.key"
        assert orch._start_live_stt(approval={}) is True
        return orch.session_info()["models"]["stt"]

    def test_Deepgram_ならそう出す(self, orch, monkeypatch):
        stt = self._start(orch, monkeypatch, "deepgram")
        assert "Deepgram" in stt["where"] and "Gemini" not in stt["where"]
        assert stt["name"] == orch._external_cfg.deepgram_model

    def test_Gemini_ならそう出す(self, orch, monkeypatch):
        stt = self._start(orch, monkeypatch, "gemini")
        assert "Gemini" in stt["where"]
        assert stt["name"] == orch._external_cfg.model

    def test_手元へ落ちたら表示も手元に戻る(self, orch, monkeypatch):
        self._start(orch, monkeypatch, "deepgram")
        orch._fall_back_to_local("試験")
        assert orch.session_info()["models"]["stt"]["where"] == "ローカル（mlx）"


class TestWhichClient:
    """この会議はどのクライアントの会議か（議事録・タスク・辞書の行き先）。

    2026-09-15: ミナトの定例の議事録が 自社 の議事録DBに入った。会議ごとに選ぶ段が無く、
    設定の `task_hub.client_name: 自社` がそのまま引き渡しに入っていた。
    """

    def test_選ばなければ自社へ引き渡す(self, orch, tmp_path):
        assert orch.client_info()["chosen"] is None
        orch._confirm_client()

        assert orch._client_name() == "自社"

    def test_選ぶと議事録もタスクも辞書もその相手になる(self, orch, monkeypatch):
        bridge = type("Bridge", (), {"set_client": lambda self, name: setattr(self, "name", name)})()
        orch._bridge = bridge
        orch._clients = [{"name": "株式会社ミナト", "client_id": "0045", "minutes_db_id": "db"}]
        events = orch.bus.subscribe()

        orch.choose_client("株式会社ミナト", "0045")

        assert orch._client_name() == "株式会社ミナト"
        assert bridge.name == "株式会社ミナト"          # タスクの行き先も同じ相手へ
        assert events.get(0.1).data["chosen"]["name"] == "株式会社ミナト"

    def test_終わりに聞いて答えが来たらそれを使う(self, orch, monkeypatch):
        """初回面談がその場で案件になることがある（運用者 指示 2026-09-16）。"""
        orch._ui_thread = object()
        orch._external_cfg.ask_timeout_sec = 5
        asked = []

        def answer(timeout):
            asked.append(timeout)
            orch.choose_client("株式会社ミナト", "0045", how="会議の終わりに画面で")
            return True

        monkeypatch.setattr(orch._client_decided, "wait", answer)
        orch._confirm_client()

        assert asked == [5] and orch._client_name() == "株式会社ミナト"

    def test_答えが無ければ自社へ倒すが黙ってはいない(self, orch, monkeypatch, capsys):
        orch._ui_thread = object()
        orch._external_cfg.ask_timeout_sec = 0.01
        monkeypatch.setattr(orch._client_decided, "wait", lambda timeout: False)

        orch._confirm_client()

        assert orch._client_name() == "自社"
        assert "自社（自社）として引き渡します" in capsys.readouterr().out

    def test_画面には自社とテストの並びで出す(self, orch):
        orch._clients = [{"name": "sample株式会社", "client_id": "sample", "minutes_db_id": ""},
                         {"name": "株式会社ミナト", "client_id": "0045", "minutes_db_id": "db"},
                         {"name": "自社", "client_id": "0000", "minutes_db_id": "db"}]

        assert [entry["name"] for entry in orch.client_info()["options"]] == [
            "自社", "株式会社ミナト", "sample株式会社"]


class TestExternalReadiness:
    """会議が始まる前に「外へ出す準備ができているか」を見る。

    2026-09-14 に踏んだ: gcloud の認証は黙って切れる。切れていると関門が
    「確認できない＝送らない」に倒れ、画面で「外へ出す」を選んでも手元に落ちる
    （会議は失われないが、画面が 17 分遅れる。その場では気づきにくい）。
    """

    def _external(self, orch, **values):
        orch._stt_engine = values.pop("stt_engine", "gemini")
        orch._llm_engine = values.pop("llm_engine", "local")
        orch._external_cfg.enabled = values.pop("enabled", True)
        orch._external_cfg.billing_project = values.pop("billing_project", "your-gcp-project")
        orch._external_cfg.key_file = values.pop("key_file", "")
        orch._external_cfg.auto_login = values.pop("auto_login", False)   # テストで本物のブラウザを開かない

    def test_準備ができていれば黙っている(self, orch, monkeypatch):
        self._external(orch)
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: (True, "課金が有効です"))

        assert orch.external_readiness() == []

    def test_認証が切れていたら直し方まで出す(self, orch, monkeypatch):
        self._external(orch)
        monkeypatch.setattr(mo, "billing_enabled",
                            lambda project, **kwargs: (False, "gcloud の認証が切れています（端末で `gcloud auth login`）"))

        warnings = orch.external_readiness()

        assert len(warnings) == 1
        assert "gcloud auth login" in warnings[0]
        assert "17 分ほど遅れます" in warnings[0]     # 何が起きるかも言う

    def test_認証が切れていたらログイン画面を開いて待ち直す(self, orch, monkeypatch):
        """端末で `gcloud auth login` を打たなくていいように（運用者 依頼 2026-09-15）。"""
        self._external(orch, auto_login=True)
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: (False, mo.LOGIN_EXPIRED))
        opened = []
        monkeypatch.setattr(mo, "login_and_wait", lambda project, **kwargs: (
            opened.append(project), kwargs["on_status"]("Google のログイン画面をブラウザで開きました"), (True, "課金が有効です"))[-1])
        events = orch.bus.subscribe()

        assert orch.external_readiness() == []
        assert opened == ["your-gcp-project"]
        assert "ログイン画面" in events.get(0.1).data["warnings"][0]     # 画面にも出す

    def test_ログインが終わらなければ警告して手元で始める(self, orch, monkeypatch):
        self._external(orch, auto_login=True)
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: (False, mo.LOGIN_EXPIRED))
        monkeypatch.setattr(mo, "login_and_wait", lambda project, **kwargs: (False, "ログインが 180 秒で終わりませんでした"))

        assert "180 秒で終わりませんでした" in orch.external_readiness()[0]

    def test_外へ出さない設定なら点検もしない(self, orch, monkeypatch):
        """手元だけで回す日に、gcloud を触りにいって待たされるのは無駄。"""
        called = []
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: called.append(project) or (True, ""))
        self._external(orch, stt_engine="local", llm_engine="local")

        assert orch.external_readiness() == []
        assert called == []

    def test_設定が食い違っていたら言う(self, orch, monkeypatch):
        """外で回す設定なのに外部送信が切ってある＝黙って手元に落ちる組み合わせ。"""
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: (True, ""))
        self._external(orch, enabled=False)

        assert "外部送信が切ってあります" in orch.external_readiness()[0]

    def test_キーが無ければ言う(self, orch, monkeypatch, tmp_path):
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: (True, ""))
        self._external(orch, key_file=str(tmp_path / "無い.key"))

        assert "API キーのファイルがありません" in orch.external_readiness()[0]

    def test_要約だけ外でも点検する(self, orch, monkeypatch):
        monkeypatch.setattr(mo, "billing_enabled", lambda project, **kwargs: (False, "課金が有効ではありません"))
        self._external(orch, stt_engine="local", llm_engine="gemini")

        warning = orch.external_readiness()[0]
        assert warning.startswith("要約を外へ出す準備ができていません")
        assert "手元の Ollama" in warning        # 何が起きるかを、その設定に合わせて言う
        assert "17 分" not in warning            # 文字起こしは手元のままなので関係ない


class TestTheClockOnEachLine:
    """行に入る時刻は「話された時刻」（届いた時刻ではない）。

    2026-09-14 の会議で実際に起きた: 会議中の文字起こしは 30 秒の窓ごとにまとめて届くので、
    届いた時刻を入れると**1 窓ぶん（10 行以上）が全部同じ時刻**になり、画面の並びも会話も
    噛み合わなくなった。記録（transcripts.jsonl）にもその時刻が残る。
    """

    def test_話された時刻が入る(self, orch):
        orch._capture_started_at = 1_757_000_000.0      # 録音の先頭（start_time の 0）
        orch._emit_segments([
            TranscriptSegment("田中", "さきの発言", 10.0, 12.0, "2020-01-01T00:00:00"),
            TranscriptSegment("田中", "あとの発言", 40.0, 42.0, "2020-01-01T00:00:00"),
        ])

        stamps = [segment.timestamp for segment in orch._transcript_buffer]
        assert stamps[0] != stamps[1]                    # 同じ窓で届いても、時刻は別
        gap = datetime.fromisoformat(stamps[1]) - datetime.fromisoformat(stamps[0])
        assert gap.total_seconds() == 30.0               # 経過の差がそのまま出る

    def test_録音が始まる前は触らない(self, orch):
        """セルフチェックの前など、まだ時計の基準が無いときは元のまま。"""
        orch._capture_started_at = None
        orch._emit_segments([TranscriptSegment("田中", "発言", 1.0, 2.0, "そのまま")])

        assert orch._transcript_buffer[0].timestamp == "そのまま"

    def test_画面へ渡す開始時刻は録音の先頭(self, orch):
        orch._capture_started_at = 1_757_000_000.0
        info = orch.session_info()

        assert datetime.fromisoformat(info["started_at"]).timestamp() == 1_757_000_000.0


class TestRenameSticksToTheVoice:
    """画面で付けた名前が、**声の主に結び付く**こと。

    2026-09-14 の会議で起きた: 「不明話者1 → NEXIS参加者D」と付けた 736 行が、作り直すと
    「不明話者1」へ戻り、2 つのクラスタに割れた。画面の表示だけが変わっていて、声紋も
    speakers.json も名前を知らなかった（`不明話者1` は「判定できなかったときの置き名」で、
    声紋が覚えた話者ではなかった）。
    """

    def _row(self, orch, start, end, amplitude=0.5, speaker="不明話者1"):
        segment = TranscriptSegment(speaker, "発言", start, end, "")
        orch._remember_voice(segment, np.array([amplitude], dtype=np.float32))
        return segment

    def test_声紋が知らない名前でも声を覚える(self, orch, tmp_path):
        self._row(orch, 10.0, 14.0)         # 4 秒の行（錨に使える長さ）
        anchored = []
        orch.diarizer.add_anchor = lambda name, embedding: anchored.append((name, embedding))

        orch.rename_speaker("不明話者1", "NEXIS参加者D")

        assert [name for name, _ in anchored] == ["NEXIS参加者D"]
        # 記録も残る（作り直しで当て直すため）
        line = json.loads((tmp_path / mo.RENAMES_FILE).read_text(encoding="utf-8").splitlines()[0])
        assert (line["old"], line["new"]) == ("不明話者1", "NEXIS参加者D")

    def test_短い行は錨にしない(self, orch):
        self._row(orch, 10.0, 10.5)         # 0.5 秒（anchor_min_sec 未満）
        anchored = []
        orch.diarizer.add_anchor = lambda name, embedding: anchored.append(name)

        orch.rename_speaker("不明話者1", "NEXIS参加者D")

        assert anchored == []               # 足せる声が無ければ足さない（誤った錨を作らない）

    def test_長いほうから数件だけ使う(self, orch):
        for index in range(8):
            self._row(orch, 100.0 + index * 10, 100.0 + index * 10 + 2.0 + index)
        anchored = []
        orch.diarizer.add_anchor = lambda name, embedding: anchored.append(name)

        orch.rename_speaker("不明話者1", "NEXIS参加者D")

        assert len(anchored) == orch.meeting.rename_anchor_rows   # 既定 5 件


def test_作り直しで画面の名前を当て直す():
    """声紋に結び付かないまま付けた名前は、記録からしか戻らない。"""
    import sys
    sys.path.insert(0, str(mo.Path(__file__).resolve().parent.parent / "scripts"))
    from finalize_meeting import apply_renames

    rows = [{"speaker": "不明話者1", "text": "あ"}, {"speaker": "自分", "text": "い"},
            {"speaker": "不明話者2", "text": "う"}]
    applied = apply_renames(rows, [{"old": "不明話者1", "new": "参加者D"},
                                   {"old": "参加者D", "new": "NEXIS参加者D"}])

    assert applied == 1
    assert [row["speaker"] for row in rows] == ["NEXIS参加者D", "自分", "不明話者2"]
