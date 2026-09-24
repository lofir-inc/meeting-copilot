"""会議中の 30 秒刻み（方式④）の窓の作り方と、時刻の戻し方。

ここで守りたいのは 2 つだけ。
  1. 切れ目は VAD の無音に置く＝語の途中で切らない（窓は溜めたチャンク単位で締まる）
  2. **無音を落として繋ぐ**ので、返ってきた時刻はそのままでは会議の時刻にならない。
     `spans` で必ず戻す（ここがずれると、話者の照合も議事録の並びも狂う）
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.vad import AudioChunk
from src.stt.live_batch import (
    AudioWindow,
    LiveBatchConfig,
    LiveBatchStt,
    Span,
    WindowAccumulator,
    config_from,
    rows_in_meeting_time,
)

SR = 16000


def chunk(start: float, end: float, value: float = 0.5) -> AudioChunk:
    samples = int((end - start) * SR)
    return AudioChunk(speaker="remote", audio=np.full(samples, value, dtype=np.float32),
                      start_time=start, end_time=end, sample_rate=SR)


def test_window_closes_when_span_reaches_the_limit():
    accumulator = WindowAccumulator("remote", SR, LiveBatchConfig(window_sec=30.0, first_window_sec=30.0))
    assert accumulator.feed(chunk(0.0, 5.0), now=100.0) is None
    assert accumulator.feed(chunk(10.0, 15.0), now=110.0) is None
    window = accumulator.feed(chunk(28.0, 31.0), now=128.0)
    assert window is not None
    # 無音（5〜10 秒・15〜28 秒）は送らない＝繋いだ長さは発話ぶんだけ
    assert window.speech_sec == pytest.approx(13.0, abs=0.01)
    assert window.start_time == 0.0 and window.end_time == 31.0
    assert accumulator.feed(chunk(32.0, 33.0), now=132.0) is None   # 締めたあとは空から


def test_long_monologue_is_capped_by_speech_length():
    accumulator = WindowAccumulator("remote", SR,
                                    LiveBatchConfig(window_sec=600.0, max_speech_sec=20.0, first_window_sec=600.0))
    assert accumulator.feed(chunk(0.0, 10.0)) is None
    assert accumulator.feed(chunk(10.0, 20.0)) is not None


def test_idle_flush_sends_what_is_left_when_the_sound_stops():
    accumulator = WindowAccumulator("remote", SR, LiveBatchConfig(window_sec=30.0, idle_flush_sec=45.0))
    accumulator.feed(chunk(0.0, 3.0), now=100.0)
    assert accumulator.tick(now=120.0) is None
    window = accumulator.tick(now=150.0)
    assert window is not None and window.end_time == 3.0
    assert accumulator.tick(now=200.0) is None   # 空なら何も出ない


def test_times_come_back_to_the_meeting_clock():
    """繋いだ音声の中の秒 → 会議の時刻。ここがずれると議事録の並びが狂う。"""
    window = AudioWindow(key="remote", audio=np.zeros(13 * SR, dtype=np.float32), sample_rate=SR,
                         spans=[Span(buffer_start=0.0, start_time=0.0, duration=5.0),
                                Span(buffer_start=5.0, start_time=10.0, duration=5.0),
                                Span(buffer_start=10.0, start_time=28.0, duration=3.0)])
    assert window.to_meeting_time(0.0) == 0.0
    assert window.to_meeting_time(4.5) == pytest.approx(4.5)
    assert window.to_meeting_time(5.0) == pytest.approx(10.0)     # 無音を跨ぐ
    assert window.to_meeting_time(12.0) == pytest.approx(30.0)
    assert window.to_meeting_time(99.0) == pytest.approx(31.0)    # 末尾を超えたら終わりで止める


def test_slice_pulls_the_audio_for_a_row_across_the_gap():
    audio = np.concatenate([np.full(5 * SR, 0.1, dtype=np.float32),
                            np.full(5 * SR, 0.9, dtype=np.float32)])
    window = AudioWindow(key="remote", audio=audio, sample_rate=SR,
                         spans=[Span(0.0, 0.0, 5.0), Span(5.0, 10.0, 5.0)])
    piece = window.slice(3.0, 12.0)
    assert piece.size == 4 * SR   # 3〜5 秒（2 秒）＋ 10〜12 秒（2 秒）。無音は含まない
    assert piece[0] == pytest.approx(0.1) and piece[-1] == pytest.approx(0.9)
    assert window.slice(6.0, 9.0).size == 0   # 送っていない無音の区間


def test_区間の境目では行を分ける():
    """無音を落として繋いで送るので、外の側では話者交代の間が消えている。

    2026-09-13 の通し確認で、別の人の「なるほど。」が前の行に吸い込まれた。
    区間の境目は元の音声では無音だった場所なので、そこで必ず切る。
    """
    window = AudioWindow(key="remote", audio=np.zeros(4 * SR, dtype=np.float32), sample_rate=SR,
                         spans=[Span(0.0, 0.0, 2.0), Span(2.0, 8.0, 2.0)])
    rows = rows_in_meeting_time(window, [], [
        {"text": "試して", "start": 0.0, "end": 1.0},
        {"text": "みました。", "start": 1.0, "end": 2.0},   # ここまでが 1 人目
        {"text": "なるほど。", "start": 2.0, "end": 3.0},   # 繋ぎ目の向こう＝別の人
    ])
    assert [row["text"] for row in rows] == ["試してみました。", "なるほど。"]
    assert rows[1]["start_time"] == pytest.approx(8.0)     # 会議の時計では 6 秒あいている


def test_同じ区間の中は間が詰まっていればつなぐ():
    window = AudioWindow(key="remote", audio=np.zeros(4 * SR, dtype=np.float32), sample_rate=SR,
                         spans=[Span(0.0, 100.0, 4.0)])
    rows = rows_in_meeting_time(window, [], [
        {"text": "そこは", "start": 0.0, "end": 0.5},
        {"text": "来週まで", "start": 0.6, "end": 1.2},
        {"text": "です。", "start": 3.0, "end": 3.5},       # 1.8 秒あいた＝切る
    ])
    assert [row["text"] for row in rows] == ["そこは来週まで", "です。"]


def test_rows_are_mapped_and_empty_text_is_dropped():
    window = AudioWindow(key="remote", audio=np.zeros(10 * SR, dtype=np.float32), sample_rate=SR,
                         spans=[Span(0.0, 100.0, 5.0), Span(5.0, 110.0, 5.0)])
    rows = rows_in_meeting_time(window, [
        {"text": "  ", "start": 0.0, "end": 1.0},
        {"text": "あとの発話", "start": 6.0, "end": 7.0},
        {"text": "さきの発話", "start": 1.0, "end": 2.0},
    ])
    assert [row["text"] for row in rows] == ["さきの発話", "あとの発話"]
    assert rows[0]["start_time"] == pytest.approx(101.0)
    assert rows[1]["start_time"] == pytest.approx(111.0)


class _FakeApi:
    """送らない口（テスト用）。`calls` に渡された音声の長さを残す。"""

    def __init__(self, utterances=None, error: Exception | None = None) -> None:
        self.calls: list[float] = []
        self._utterances = utterances or []
        self._error = error

    def transcribe_audio(self, audio, sample_rate, work_dir):
        self.calls.append(audio.size / sample_rate)
        if self._error is not None:
            raise self._error
        return {"utterances": self._utterances, "usage": {"promptTokenCount": 10, "candidatesTokenCount": 2}}


def test_results_come_back_in_meeting_time(tmp_path):
    api = _FakeApi([{"text": "そこは決めましょう", "start": 0.5, "end": 2.0}])
    got: list[tuple[str, list[dict]]] = []
    stt = LiveBatchStt(api, tmp_path, lambda key, window, rows: got.append((key, rows)),
                       config=LiveBatchConfig(window_sec=5.0, first_window_sec=5.0), sample_rate=SR, keys=("remote",))
    stt.feed("remote", chunk(20.0, 26.0))
    stt.close(wait=True)
    assert api.calls == [pytest.approx(6.0)]
    assert got and got[0][0] == "remote"
    assert got[0][1][0]["start_time"] == pytest.approx(20.5)
    assert stt.usage.windows == 1 and stt.usage.prompt_tokens == 10


def test_a_failed_window_is_handed_over_as_a_gap(tmp_path):
    api = _FakeApi(error=RuntimeError("HTTP 503"))
    gaps: list[tuple[float, float]] = []
    stt = LiveBatchStt(api, tmp_path, lambda key, window, rows: None,
                       config=LiveBatchConfig(window_sec=5.0, first_window_sec=5.0, retries=1, retry_wait_sec=0.0),
                       sample_rate=SR, keys=("remote",),
                       on_gap=lambda key, window, error: gaps.append((window.start_time, window.end_time)),
                       sleep=lambda seconds: None)
    stt.feed("remote", chunk(0.0, 6.0))
    stt.close(wait=True)
    assert len(api.calls) == 2          # 1 回送り直してから諦める
    assert gaps == [(0.0, 6.0)]         # 取りこぼした区間は呼び出し側へ渡る
    assert stt.usage.gaps == 1 and stt.pending == 0


def test_後ろの窓が先に返っても順番は入れ替わらない(tmp_path):
    """送り直しが入ると後ろの窓が先に返る。順番が狂うと話者の照合と画面の並びが狂う。"""
    import threading

    gate = threading.Event()

    class SlowFirst:
        """さきの窓（振幅 0.1）だけ待たせる口。音で見分ける（呼ばれた数だと競合する）。"""

        def transcribe_audio(self, audio, sample_rate, work_dir):
            first = float(np.max(np.abs(audio))) < 0.5
            if first:
                gate.wait(2.0)
            return {"utterances": [{"text": "さきの窓" if first else "あとの窓",
                                    "start": 0.0, "end": 1.0}], "usage": {}}

    got: list[str] = []
    stt = LiveBatchStt(SlowFirst(), tmp_path, lambda key, window, rows: got.append(rows[0]["text"]),
                       config=LiveBatchConfig(window_sec=5.0, first_window_sec=5.0, max_workers=2), sample_rate=SR,
                       keys=("remote",))
    stt.feed("remote", chunk(0.0, 6.0, value=0.1))     # さきの窓（待たされる）
    stt.feed("remote", chunk(10.0, 16.0, value=0.9))   # あとの窓（先に返る）
    gate.set()
    stt.close(wait=True)

    assert got == ["さきの窓", "あとの窓"]      # 送った順にだけ渡る


def test_送れなかった窓は順番を止めない(tmp_path):
    """失敗した窓で後続が詰まってはいけない（会議中の画面が止まる）。"""
    class FirstFails:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe_audio(self, audio, sample_rate, work_dir):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("HTTP 503")
            return {"utterances": [{"text": "あとの窓", "start": 0.0, "end": 1.0}], "usage": {}}

    got: list[str] = []
    stt = LiveBatchStt(FirstFails(), tmp_path, lambda key, window, rows: got.append(rows[0]["text"]),
                       config=LiveBatchConfig(window_sec=5.0, first_window_sec=5.0, retries=0), sample_rate=SR,
                       keys=("remote",), on_gap=lambda key, window, error: None)
    stt.feed("remote", chunk(0.0, 6.0))
    stt.close(wait=True)
    stt = None
    assert got == []

    # 2 本目だけを流す（1 本目は取りこぼしとして記録済み）
    api = FirstFails()
    api.calls = 1
    got2: list[str] = []
    stt2 = LiveBatchStt(api, tmp_path, lambda key, window, rows: got2.append(rows[0]["text"]),
                        config=LiveBatchConfig(window_sec=5.0, first_window_sec=5.0), sample_rate=SR, keys=("remote",))
    stt2.feed("remote", chunk(10.0, 16.0))
    stt2.close(wait=True)
    assert got2 == ["あとの窓"]


def test_最初の窓だけ短く締める():
    """「最初にすぐ出て来ないのが不安」（2026-09-14・会議前の点検で 運用者）。

    30 秒刻みだと最初の 1 行まで 40 秒前後かかり、録れているのかどうかが分からない。
    最初だけ 10 秒で締めれば 20 秒ほどで 1 行出る。2 本目からは 30 秒に戻す（費用も精度も変えない）。
    """
    accumulator = WindowAccumulator("remote", SR, LiveBatchConfig(window_sec=30.0, first_window_sec=10.0))

    assert accumulator.feed(chunk(0.0, 4.0), now=100.0) is None
    first = accumulator.feed(chunk(5.0, 11.0), now=106.0)
    assert first is not None and first.end_time == 11.0        # 最初は 10 秒で締まる

    assert accumulator.feed(chunk(12.0, 20.0), now=112.0) is None   # 2 本目は 30 秒まで待つ
    assert accumulator.feed(chunk(21.0, 30.0), now=121.0) is None
    second = accumulator.feed(chunk(31.0, 43.0), now=131.0)
    assert second is not None and second.start_time == 12.0


class TestConfigFrom:
    """設定（`meeting.external_stt.live`）から刻み方を組む。

    `auto` は「エンジンに合わせる」の意味で、**数ではない**。
      2026-09-20 まで、この解決は会議側にしか無く、`scripts/selftest_live.py` は
      見本の設定（`window_sec: auto`）のまま動かすと
      `TypeError: '>=' not supported between 'float' and 'str'` で必ず落ちた。
      SETUP がそのコマンドを案内しているので、配布先は全員踏む。
    """

    def test_autoはエンジンの既定になる(self):
        deepgram = config_from("deepgram", {"window_sec": "auto", "first_window_sec": "auto"})
        gemini = config_from("gemini", {"window_sec": "auto"})
        assert isinstance(deepgram.window_sec, float) and isinstance(gemini.window_sec, float)
        assert deepgram.window_sec < gemini.window_sec      # Deepgram は往復が速いので短い

    def test_空と_none_も同じ扱い(self):
        for value in ("", "none", "  AUTO  "):
            assert isinstance(config_from("deepgram", {"window_sec": value}).window_sec, float)

    def test_数が書いてあればそれが勝つ(self):
        assert config_from("deepgram", {"window_sec": 12}).window_sec == 12

    def test_知らないキーは無視する(self):
        assert config_from("deepgram", {"kakunai": 1}).window_sec > 0

    def test_設定が空でも組める(self):
        assert config_from("gemini", None).window_sec > 0
