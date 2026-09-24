"""ScreenCaptureKit 再ブロッカの単体テスト。"""

import numpy as np

from src.audio.sck_capture import _Reblocker


def test_five_20ms_buffers_make_one_100ms_frame():
    reblocker = _Reblocker(1000, 0.1)
    frames = []
    for index in range(5):
        frames.extend(reblocker.push(np.full(20, index, dtype=np.float32), index * 0.02))
    assert len(frames) == 1
    assert frames[0].shape == (100,)
    np.testing.assert_array_equal(frames[0], np.repeat(np.arange(5, dtype=np.float32), 20))


def test_pts_gap_inserts_silence_and_tracks_gap_statistics():
    reblocker = _Reblocker(1000, 0.1)
    reblocker.push(np.ones(20, dtype=np.float32), 0.0)
    frames = reblocker.push(np.ones(20, dtype=np.float32) * 2, 0.52)
    combined = np.concatenate(frames + [reblocker.flush()])
    assert np.count_nonzero(combined[20:520]) == 0
    assert reblocker.gap_max_sec == 0.5
    assert reblocker.gap_total_sec == 0.5


def test_partial_frame_is_carried_to_the_next_push():
    reblocker = _Reblocker(1000, 0.1)
    assert reblocker.push(np.ones(60, dtype=np.float32), 0.0) == []
    frames = reblocker.push(np.ones(40, dtype=np.float32) * 2, 0.06)
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0][:60], np.ones(60, dtype=np.float32))
    np.testing.assert_array_equal(frames[0][60:], np.ones(40, dtype=np.float32) * 2)


def test_flush_returns_remaining_partial_frame():
    reblocker = _Reblocker(1000, 0.1)
    reblocker.push(np.ones(30, dtype=np.float32), 0.0)
    remaining = reblocker.flush()
    assert remaining is not None
    assert remaining.shape == (30,)
    assert reblocker.flush() is None


def test_continuous_pts_does_not_create_a_gap():
    reblocker = _Reblocker(1000, 0.1)
    reblocker.push(np.ones(20, dtype=np.float32), 0.0)
    reblocker.push(np.ones(20, dtype=np.float32), 0.02)
    assert reblocker.gap_total_sec == 0
    assert reblocker.gap_max_sec == 0
