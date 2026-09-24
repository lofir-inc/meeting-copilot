"""複数入力デバイスの同時キャプチャの単体テスト。

実デバイスを開かないよう `sd.InputStream` を差し替える。
"""

import wave

import numpy as np
import pytest

from src.audio import multi_capture as mc
from src.audio.multi_capture import MultiAudioConfig, MultiCapture, SourceConfig, to_mono


class FakeStream:
    """sd.InputStream の代役。callback を握っておいて手動で叩けるようにする。"""

    instances: list["FakeStream"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def emit(self, frames: np.ndarray) -> None:
        self.callback(frames, len(frames), None, None)


@pytest.fixture
def fake_streams(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setattr(mc.sd, "InputStream", FakeStream)
    return FakeStream


@pytest.fixture
def sources():
    return [
        SourceConfig(key="self", device=6, device_name="Headset One", channels=1, speaker="自分"),
        SourceConfig(key="remote", device=8, device_name="BlackHole 2ch", channels=2, speaker=None),
    ]


class TestToMono:
    def test_mono_input_passes_through(self):
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_almost_equal(to_mono(data), data)

    def test_single_channel_2d_is_flattened(self):
        data = np.array([[0.1], [0.2]], dtype=np.float32)
        assert to_mono(data).shape == (2,)

    def test_stereo_is_averaged(self):
        data = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(to_mono(data), [0.5, 0.5])

    def test_returns_a_copy(self):
        data = np.array([0.1, 0.2], dtype=np.float32)
        out = to_mono(data)
        out[0] = 9.0
        assert data[0] == pytest.approx(0.1)


class TestConstruction:
    def test_opens_one_stream_per_source(self, fake_streams, sources):
        MultiCapture(sources, MultiAudioConfig())
        assert len(fake_streams.instances) == 2
        assert [s.kwargs["device"] for s in fake_streams.instances] == [6, 8]

    def test_channel_count_follows_the_device(self, fake_streams, sources):
        MultiCapture(sources, MultiAudioConfig())
        assert [s.kwargs["channels"] for s in fake_streams.instances] == [1, 2]

    def test_rejects_empty_sources(self, fake_streams):
        with pytest.raises(ValueError):
            MultiCapture([], MultiAudioConfig())

    def test_rejects_duplicate_keys(self, fake_streams, sources):
        dup = [sources[0], SourceConfig(key="self", device=9, device_name="x", channels=1)]
        with pytest.raises(ValueError):
            MultiCapture(dup, MultiAudioConfig())


class TestQueues:
    def test_streams_stay_separate(self, fake_streams, sources):
        """ここが本丸。ループバックに自分の声は入らないので、混ぜてはいけない。"""
        capture = MultiCapture(sources, MultiAudioConfig())
        capture.start()

        mic_stream, loopback_stream = fake_streams.instances
        mic_stream.emit(np.full((4800, 1), 0.3, dtype=np.float32))
        loopback_stream.emit(np.full((4800, 2), 0.7, dtype=np.float32))

        mic_frames = capture.drain("self")
        remote_frames = capture.drain("remote")

        assert len(mic_frames) == 1 and len(remote_frames) == 1
        assert mic_frames[0][0] == pytest.approx(0.3)
        assert remote_frames[0][0] == pytest.approx(0.7)

    def test_drain_empties_the_queue(self, fake_streams, sources):
        capture = MultiCapture(sources, MultiAudioConfig())
        fake_streams.instances[0].emit(np.zeros((480, 1), dtype=np.float32))
        assert len(capture.drain("self")) == 1
        assert capture.drain("self") == []

    def test_stereo_loopback_is_stored_as_mono(self, fake_streams, sources):
        capture = MultiCapture(sources, MultiAudioConfig())
        fake_streams.instances[1].emit(np.zeros((480, 2), dtype=np.float32))
        assert capture.drain("remote")[0].ndim == 1


class TestStartOffsets:
    def test_earliest_stream_is_zero(self, fake_streams, sources):
        capture = MultiCapture(sources, MultiAudioConfig())
        fake_streams.instances[0].emit(np.zeros((480, 1), dtype=np.float32))
        fake_streams.instances[1].emit(np.zeros((480, 2), dtype=np.float32))

        offsets = capture.start_offsets()
        assert set(offsets) == {"self", "remote"}
        assert min(offsets.values()) == pytest.approx(0.0)
        assert offsets["self"] == pytest.approx(0.0)   # 先に届いた方が基準
        assert offsets["remote"] >= 0.0

    def test_offsets_measure_the_gap(self, fake_streams, sources, monkeypatch):
        clock = iter([100.0, 100.25])
        monkeypatch.setattr(mc.time, "time", lambda: next(clock))

        capture = MultiCapture(sources, MultiAudioConfig())
        fake_streams.instances[0].emit(np.zeros((480, 1), dtype=np.float32))
        fake_streams.instances[1].emit(np.zeros((480, 2), dtype=np.float32))

        offsets = capture.start_offsets()
        assert offsets["self"] == pytest.approx(0.0)
        assert offsets["remote"] == pytest.approx(0.25)

    def test_no_frames_yet(self, fake_streams, sources):
        capture = MultiCapture(sources, MultiAudioConfig())
        assert capture.start_offsets() == {}

    def test_wait_returns_false_when_a_source_is_dead(self, fake_streams, sources):
        """ループバックだけ何も来ない＝Zoom の出力先が違う、を検出できる。"""
        capture = MultiCapture(sources, MultiAudioConfig())
        fake_streams.instances[0].emit(np.zeros((480, 1), dtype=np.float32))

        assert not capture.wait_for_first_frames(timeout=0.05)
        assert capture.silent_sources() == ["remote"]

    def test_wait_returns_true_when_all_arrive(self, fake_streams, sources):
        capture = MultiCapture(sources, MultiAudioConfig())
        fake_streams.instances[0].emit(np.zeros((480, 1), dtype=np.float32))
        fake_streams.instances[1].emit(np.zeros((480, 2), dtype=np.float32))

        assert capture.wait_for_first_frames(timeout=0.05)
        assert capture.silent_sources() == []


class TestRecording:
    def test_writes_one_wav_per_source(self, fake_streams, sources, tmp_path):
        capture = MultiCapture(sources, MultiAudioConfig(), record_dir=tmp_path)
        fake_streams.instances[0].emit(np.full((4800, 1), 0.5, dtype=np.float32))
        fake_streams.instances[1].emit(np.full((4800, 2), 0.5, dtype=np.float32))
        capture.stop()

        for key in ("self", "remote"):
            path = tmp_path / f"recording_{key}.wav"
            assert path.exists()
            with wave.open(str(path)) as w:
                assert w.getnchannels() == 1
                assert w.getframerate() == 48000
                assert w.getnframes() == 4800

    def test_stop_closes_every_stream(self, fake_streams, sources):
        capture = MultiCapture(sources, MultiAudioConfig())
        capture.start()
        capture.stop()
        assert all(s.closed for s in fake_streams.instances)


class TestRecordingIsNotOverwritten:
    def test_previous_recording_is_kept_on_restart(self, fake_streams, sources, tmp_path):
        """2026-09-11 本番: 同じセッションで立ち上げ直すと、前の録音を頭から書き潰すところだった。"""
        old = tmp_path / "recording_self.wav"
        old.write_bytes(b"previous meeting audio")
        MultiCapture(sources, MultiAudioConfig(), record_dir=tmp_path)
        kept = sorted(tmp_path.glob("recording_self.*.wav"))
        assert len(kept) == 1
        assert kept[0].read_bytes() == b"previous meeting audio"
        assert (tmp_path / "recording_self.wav").exists()

    def test_fresh_session_has_no_backup(self, fake_streams, sources, tmp_path):
        MultiCapture(sources, MultiAudioConfig(), record_dir=tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.wav")) == ["recording_remote.wav", "recording_self.wav"]


class TestTakenSeconds:
    def test_drain_and_record_from_are_counted(self, fake_streams, sources):
        cap = MultiCapture(sources, MultiAudioConfig(sample_rate=48000))
        cap.start()
        mic = fake_streams.instances[0]
        mic.emit(np.zeros((48000, 1), dtype=np.float32))
        cap.drain("self")
        assert cap.taken_seconds("self") == pytest.approx(1.0)
        mic.emit(np.zeros((24000, 1), dtype=np.float32))
        cap.record_from("self", 0.5, discard_backlog=False)
        assert cap.taken_seconds("self") == pytest.approx(1.5)
        assert cap.taken_seconds("remote") == 0.0


class TestTrimBacklog:
    """本編前の挨拶を文字起こしに残す（運用者 の運用フロー 2026-09-12）。"""

    def _emit(self, mic, seconds: float, sample_rate: int = 48000) -> None:
        for _ in range(int(seconds * 10)):          # 0.1 秒フレーム
            mic.emit(np.zeros((sample_rate // 10, 1), dtype=np.float32))

    def test_recent_audio_is_kept_and_old_audio_is_dropped(self, fake_streams, sources, tmp_path):
        cap = MultiCapture(sources, MultiAudioConfig(sample_rate=48000))
        cap.start()
        self._emit(fake_streams.instances[0], 10.0)

        dropped = cap.trim_backlog("self", keep_seconds=3.0)

        assert dropped == pytest.approx(7.0, abs=0.2)
        remaining = cap.drain("self")
        assert sum(len(frame) for frame in remaining) / 48000 == pytest.approx(3.0, abs=0.2)

    def test_kept_audio_is_not_counted_twice_for_clock_alignment(self, fake_streams, sources):
        """戻した分を taken に数えると、文字起こしの時刻が録音からずれる。"""
        cap = MultiCapture(sources, MultiAudioConfig(sample_rate=48000))
        cap.start()
        self._emit(fake_streams.instances[0], 10.0)

        cap.trim_backlog("self", keep_seconds=3.0)

        assert cap.taken_seconds("self") == pytest.approx(7.0, abs=0.2)

    def test_short_backlog_is_kept_whole(self, fake_streams, sources):
        cap = MultiCapture(sources, MultiAudioConfig(sample_rate=48000))
        cap.start()
        self._emit(fake_streams.instances[0], 2.0)

        assert cap.trim_backlog("self", keep_seconds=3.0) == 0.0
        assert cap.taken_seconds("self") == 0.0
