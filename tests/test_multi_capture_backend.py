"""MultiCapture の ScreenCaptureKit バックエンド配線と、record_from / drift の単体テスト。

SCK 本体は実機でしか動かないので SckSystemAudio をフェイクに差し替える。
"""

import numpy as np
import pytest

from src.audio import multi_capture as mc
from src.audio.multi_capture import MultiAudioConfig, MultiCapture, SourceConfig

SR = 48000


class FakeSck:
    instances: list["FakeSck"] = []

    def __init__(self, cfg, on_frame):
        self.cfg = cfg
        self.on_frame = on_frame
        self.started = False
        FakeSck.instances.append(self)

    def start(self, timeout=5.0):
        self.started = True

    def stop(self):
        self.started = False

    def stats(self):
        return {"buffers": 1}


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def capture(monkeypatch):
    FakeSck.instances = []
    monkeypatch.setattr(mc, "SckSystemAudio", FakeSck)
    monkeypatch.setattr(mc.sd, "InputStream", FakeStream)
    cap = MultiCapture(
        [
            SourceConfig("self", 6, "mic", 1, "自分"),
            SourceConfig("remote", None, "ScreenCaptureKit", 1, None, backend="screencapturekit"),
        ],
        MultiAudioConfig(sample_rate=SR, chunk_duration_sec=0.1),
    )
    cap.start()
    return cap


def test_screencapturekit_source_uses_sck_instead_of_sounddevice(capture):
    assert len(FakeSck.instances) == 1
    assert FakeSck.instances[0].started
    assert FakeSck.instances[0].cfg.sample_rate == SR
    assert FakeSck.instances[0].cfg.frame_sec == pytest.approx(0.1)


def test_sck_frames_reach_the_queue_through_ingest(capture):
    FakeSck.instances[0].on_frame(np.full(4800, 0.25, dtype=np.float32))
    frames = capture.drain("remote")
    assert len(frames) == 1
    assert frames[0].shape == (4800,)


def test_record_from_discards_backlog_by_default(capture):
    """登録より前にキャプチャが始まるので、溜まった古い音を登録音声にしない。"""
    import threading

    sck = FakeSck.instances[0]
    sck.on_frame(np.full(4800, 0.9, dtype=np.float32))   # 古い（名前入力中に溜まった）
    sck.on_frame(np.full(4800, 0.9, dtype=np.float32))
    # 呼んだ直後に「これから」の音が届く
    threading.Timer(0.05, lambda: sck.on_frame(np.full(4800, 0.1, dtype=np.float32))).start()
    audio = capture.record_from("remote", 0.1)
    assert audio.size == 4800
    assert float(audio.max()) == pytest.approx(0.1)


def test_record_from_can_keep_backlog(capture):
    sck = FakeSck.instances[0]
    sck.on_frame(np.full(4800, 0.9, dtype=np.float32))
    audio = capture.record_from("remote", 0.1, discard_backlog=False)
    assert float(audio.max()) == pytest.approx(0.9)


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(mc.sd, "InputStream", FakeStream)
    with pytest.raises(ValueError):
        MultiCapture([SourceConfig("x", 1, "x", 1, backend="coreaudio")], MultiAudioConfig(sample_rate=SR))


def test_drift_is_zero_right_after_first_frame(capture, monkeypatch):
    clock = iter([1000.0, 1000.1])   # 初回フレーム時刻 → drift() の now（0.1 秒後に 0.1 秒ぶん届いている）
    monkeypatch.setattr(mc.time, "time", lambda: next(clock))
    FakeSck.instances[0].on_frame(np.zeros(4800, dtype=np.float32))
    drift = capture.drift()
    assert drift["remote"] == pytest.approx(0.0, abs=1e-6)
