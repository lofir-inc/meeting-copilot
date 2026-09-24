"""会議中の文字起こしを、30 秒ごとのバッチで外（Gemini）に投げる（方式④＝準リアルタイム・バッチ）。

なぜこの形か（2026-09-13 の実測。詳しくは `PLAN-stt-selection.md`）

| 方式 | 取りこぼし | 画面に出るまで | GPU | 月額 |
|---|---|---|---|---|
| ローカル・会議中（従来） | 6.4% | **17 分遅れ** | 2 つ | ¥0 |
| Gemini Live（ストリーミング） | 3.6% | 0.3 秒 | 1 つ | ¥2,023〜3,540 |
| **ここ（30 秒刻みのバッチ）** | **0.5%** | 28 秒 | 1 つ | ¥759〜1,600 |

バッチのモデルを短く刻んでも精度は落ちない（30 秒刻みと 25 分ひと刻みを 2 区間で比較して確認）。
Live と違ってセッションの 9 分上限が無く、費用は Live の 0.56 倍。

刻み方の決まりごと
  - 切れ目は**必ず VAD の無音**に置く（語の途中で切らない）。溜めた発話をつないで送り、
    無音は送らない。無音混じりを投げると専用モデルが**何も返さない**ことがある
    （2026-09-13 実測。`scripts/gemini_transcribe.py` の `content_end` の注記）
  - つないだぶん、返ってきた時刻は**そのままでは会議の時刻にならない**。`AudioWindow.spans`
    で戻す（1 区間ごとに「送った音声の中の位置 → 会議の時刻」を持つ）
  - **話者は外に任せない**（Gemini 76% 対 手元 97%）。ここは文字だけを返し、
    誰が言ったかは `src/audio/segment_labeler.py` が手元の声紋で当てる

失敗しても会議は止めない。送れなかった窓は `on_gap` で呼び出し側へ渡し、
  会議のあとにその区間だけ拾い直す（`scripts/finalize_meeting.py --reuse-live`）。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.audio.vad import AudioChunk
from src.stt.gemini_transcribe import TranscribeApi

logger = logging.getLogger(__name__)

GAPS_FILE = "external_gaps.jsonl"
"""会議中に外へ送れなかった区間を書くファイル（セッション直下）。

会議のあとに、**ここだけ**拾い直す（`scripts/finalize_meeting.py`）。方式④は「会議後は送り直さない」
のが要点なので、拾い直す範囲はこの記録が決める。
"""


@dataclass
class LiveBatchConfig:
    """会議中のバッチ送信の刻み方。"""

    window_sec: float = 30.0
    """ここまで溜まったら送る（発話の始まりから終わりまでの幅）。

    画面に出るまで ≒ これ ＋ 往復。往復はエンジンで桁が違うので、既定は
    `defaults_for()` がエンジンごとに決める（ここの値は最後の砦）。
    """

    first_window_sec: float = 10.0
    """**最初の窓だけ**短くする。

    2026-09-14 の会議前の点検で 運用者 が言った「最初にすぐ出て来ないのが不安」。
    30 秒刻みだと最初の 1 行まで 40 秒前後かかり、**録れているのかどうかが分からない**。
    最初だけ短くすれば 20 秒ほどで 1 行出る。以後は 30 秒刻みに戻す（費用と精度は変えない）。
    """

    idle_flush_sec: float = 45.0
    """溜まったまま音が途切れたとき、壁時計でここまで待って送る。"""

    max_speech_sec: float = 120.0
    """1 回に送る音声（無音を除いた実質）の上限。長い独話で窓が膨らむのを止める。"""

    max_workers: int = 2
    """同時に送る数。2 系統（自分・相手）ぶん。レート制限は 10 リクエスト/分で、
    30 秒刻みなら 2 系統で 4 req/分。"""

    retries: int = 1
    """失敗したときに送り直す回数。これを使い切ったら窓ごと `on_gap` へ回す。"""

    retry_wait_sec: float = 3.0


DEFAULTS: dict[str, dict[str, float]] = {
    "deepgram": {"window_sec": 8.0, "first_window_sec": 4.0},
    "gemini": {"window_sec": 30.0, "first_window_sec": 10.0},
}
"""エンジンごとの刻み方。どちらも実測から決めた。

**Deepgram（往復の中央 1.5 秒・2026-09-18 の実会議 98 窓）**
窓を縮めても精度が落ちないところまで縮めた。会話 13.1 分で測った結果:

    まるごと 38.2% ／ 30 秒 39.8% ／ 15 秒 40.5% ／ **8 秒 39.8%** ／ 5 秒 43.2% ／ 3 秒 44.3%

**8 秒は 30 秒と同じ精度**で、5 秒から崩れる。∴ 8 秒。
画面に出るまで 8 ＋ 1.5 ＝ **約 9.5 秒**（30 秒のときは 31.5 秒だった）。

**Gemini（往復 12.6 秒）**
縮めても往復で戻ってくるので効き目が薄いうえ、レート制限が 10 リクエスト/分。
8 秒にすると 2 系統で 15 リクエスト/分になり、制限に当たる。∴ 30 秒のまま。

手元の Whisper はここを通らない（無音で切って最大 10 秒。`audio.max_chunk_sec`）。
"""


def defaults_for(provider: str) -> dict[str, float]:
    """そのエンジンに合った刻み方。知らないエンジンなら空（`LiveBatchConfig` の既定に任せる）。"""
    return dict(DEFAULTS.get(str(provider or "").lower(), {}))


def config_from(provider: str, told: dict | None) -> "LiveBatchConfig":
    """設定（`meeting.external_stt.live`）から刻み方を組む。

    `auto`・空・`none` は「**エンジンに合わせる**」の意味で、数ではない。
      ここを通さずに `LiveBatchConfig(**設定)` とすると、`window_sec` に文字列 "auto" が入り、
      比較したところで `TypeError: '>=' not supported between 'float' and 'str'` で落ちる。
      2026-09-20 まで、この解決は会議側にしか無く、`scripts/selftest_live.py` は
      **見本の設定のまま動かすと必ず落ちた**（SETUP がそのコマンドを案内している）。
    """
    known = set(LiveBatchConfig.__dataclass_fields__)
    values = {key: value for key, value in (told or {}).items()
              if key in known and str(value).strip().lower() not in {"auto", "", "none"}}
    return LiveBatchConfig(**{**defaults_for(provider), **values})


@dataclass
class Span:
    """送った音声の中の位置と、会議の時刻の対応。"""

    buffer_start: float
    start_time: float
    duration: float

    @property
    def buffer_end(self) -> float:
        return self.buffer_start + self.duration

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class AudioWindow:
    """1 回ぶんの送りもの（無音を落として繋いだ音声 ＋ 時刻の対応表）。"""

    key: str
    audio: np.ndarray
    sample_rate: int
    spans: list[Span] = field(default_factory=list)

    @property
    def start_time(self) -> float:
        return self.spans[0].start_time if self.spans else 0.0

    @property
    def end_time(self) -> float:
        return self.spans[-1].end_time if self.spans else 0.0

    @property
    def speech_sec(self) -> float:
        return float(self.audio.size / self.sample_rate) if self.sample_rate else 0.0

    def span_index(self, buffer_time: float) -> int:
        """送った音声の中の秒が、何番目の区間（＝VAD で切った発話）に入るか。

        区間をまたぐところは、元の音声では**無音があった場所**。繋いだ音声では消えているので、
        ここを跨いで 1 つの発話にまとめてはいけない（別の人の相づちを吸ってしまう）。
        """
        for index, span in enumerate(self.spans):
            if buffer_time < span.buffer_end:
                return index
        return max(len(self.spans) - 1, 0)

    def to_meeting_time(self, buffer_time: float) -> float:
        """送った音声の中の秒を、会議の時刻に戻す。"""
        if not self.spans:
            return float(buffer_time)
        for span in self.spans:
            if buffer_time < span.buffer_end or span is self.spans[-1]:
                inside = min(max(buffer_time - span.buffer_start, 0.0), span.duration)
                return span.start_time + inside
        return self.end_time

    def slice(self, start_time: float, end_time: float) -> np.ndarray:
        """会議の時刻で切り出す（繋いだ音声から拾い直す）。話者の照合に使う。"""
        pieces: list[np.ndarray] = []
        for span in self.spans:
            begin = max(start_time, span.start_time)
            finish = min(end_time, span.end_time)
            if finish <= begin:
                continue
            offset = span.buffer_start + (begin - span.start_time)
            pieces.append(self.audio[int(offset * self.sample_rate):int((offset + finish - begin) * self.sample_rate)])
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


class WindowAccumulator:
    """VAD のチャンクを窓ぶん溜めて、無音の切れ目でひとまとめにする。"""

    def __init__(self, key: str, sample_rate: int, config: LiveBatchConfig | None = None) -> None:
        self.key = key
        self.sample_rate = sample_rate
        self.config = config or LiveBatchConfig()
        self._chunks: list[AudioChunk] = []
        self._opened_at = 0.0
        self._closed_any = False
        """1 つでも窓を締めたか（最初の窓だけ短くするため）。"""

    @property
    def pending_sec(self) -> float:
        """溜まっている発話の長さ（会議の時刻での幅）。"""
        if not self._chunks:
            return 0.0
        return self._chunks[-1].end_time - self._chunks[0].start_time

    def feed(self, chunk: AudioChunk, now: float | None = None) -> AudioWindow | None:
        """1 チャンク溜める。窓が埋まったら `AudioWindow` を返す。"""
        if not self._chunks:
            self._opened_at = now if now is not None else time.time()
        self._chunks.append(chunk)
        speech = sum(item.end_time - item.start_time for item in self._chunks)
        # 最初の 1 回だけ短く締める（「録れているのか分からない」時間を短くするため）
        width = self.config.first_window_sec if not self._closed_any else self.config.window_sec
        if self.pending_sec >= width or speech >= self.config.max_speech_sec:
            return self.flush()
        return None

    def tick(self, now: float | None = None) -> AudioWindow | None:
        """音が途切れたまま溜まっているぶんを、壁時計で締める。"""
        now = now if now is not None else time.time()
        if self._chunks and now - self._opened_at >= self.config.idle_flush_sec:
            return self.flush()
        return None

    def flush(self) -> AudioWindow | None:
        """溜まっているぶんを窓にして返す（無ければ None）。"""
        chunks, self._chunks = self._chunks, []
        if not chunks:
            return None
        self._closed_any = True
        pieces: list[np.ndarray] = []
        spans: list[Span] = []
        position = 0.0
        for chunk in chunks:
            audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
            if not audio.size:
                continue
            duration = audio.size / self.sample_rate
            pieces.append(audio)
            spans.append(Span(buffer_start=position, start_time=chunk.start_time, duration=duration))
            position += duration
        if not pieces:
            return None
        return AudioWindow(key=self.key, sample_rate=self.sample_rate, spans=spans,
                           audio=np.concatenate(pieces) if len(pieces) > 1 else pieces[0])


@dataclass
class BatchUsage:
    """使った量（実費の見積もりと、あとからの突き合わせに使う）。"""

    windows: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    gaps: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def median_latency_sec(self) -> float:
        return float(np.median(self.latencies)) if self.latencies else 0.0


class LiveBatchStt:
    """窓を溜めて、非同期に投げて、返ってきたら呼び出し側へ渡す。"""

    def __init__(
        self,
        api: TranscribeApi,
        work_dir: Path,
        on_rows: Callable[[str, AudioWindow, list[dict]], None],
        *,
        config: LiveBatchConfig | None = None,
        sample_rate: int = 48000,
        keys: tuple[str, ...] = ("self", "remote"),
        on_gap: Callable[[str, AudioWindow, Exception], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api = api
        self.config = config or LiveBatchConfig()
        self.work_dir = Path(work_dir)
        self.usage = BatchUsage()
        self._on_rows = on_rows
        self._on_gap = on_gap
        self._sleep = sleep
        self._accumulators = {key: WindowAccumulator(key, sample_rate, self.config) for key in keys}
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers,
                                            thread_name_prefix="live-batch")
        self._pending = 0
        self._lock = threading.Lock()
        self._sent = {key: 0 for key in keys}
        """系統ごとに送った窓の番号（渡す順を揃えるため）。"""
        self._next = {key: 0 for key in keys}
        self._held: dict[str, dict[int, tuple]] = {key: {} for key in keys}
        """先に返ってきた後ろの窓を、前の窓が片付くまで預かる場所。"""

    # ------------------------------------------------------------------ 入口

    def feed(self, key: str, chunk: AudioChunk, now: float | None = None) -> None:
        """VAD のチャンクを溜める。窓が埋まればそのまま送りに出す。"""
        accumulator = self._accumulators.get(key)
        if accumulator is None:
            accumulator = self._accumulators[key] = WindowAccumulator(key, self._sample_rate(), self.config)
        window = accumulator.feed(chunk, now=now)
        if window is not None:
            self._submit(window)

    def tick(self, now: float | None = None) -> None:
        """主ループから毎回呼ぶ。音が途切れたまま溜まっている窓を締める。"""
        for accumulator in self._accumulators.values():
            window = accumulator.tick(now)
            if window is not None:
                self._submit(window)

    def flush(self) -> None:
        """溜まっているぶんを全部送りに出す（会議の終わりに呼ぶ）。"""
        for accumulator in self._accumulators.values():
            window = accumulator.flush()
            if window is not None:
                self._submit(window)

    def close(self, wait: bool = True) -> None:
        """送信中のぶんを待って畳む。"""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    @property
    def pending_sec(self) -> float:
        return max((item.pending_sec for item in self._accumulators.values()), default=0.0)

    # ------------------------------------------------------------------ 内部

    def _sample_rate(self) -> int:
        return next(iter(self._accumulators.values())).sample_rate

    def _submit(self, window: AudioWindow) -> None:
        with self._lock:
            self._pending += 1
            self._sent.setdefault(window.key, 0)
            self._held.setdefault(window.key, {})
            self._next.setdefault(window.key, 0)
            order = self._sent[window.key]
            self._sent[window.key] += 1
        self._executor.submit(self._send, window, order)

    def _send(self, window: AudioWindow, order: int) -> None:
        started = time.time()
        try:
            rows = self._transcribe(window)
        except Exception as error:  # noqa: BLE001 — 会議は止めない
            self.usage.failures += 1
            self.usage.gaps += 1
            logger.warning("外へ送れませんでした（%s %.1f〜%.1f 秒）: %s",
                           window.key, window.start_time, window.end_time, error)
            if self._on_gap is not None:
                try:
                    self._on_gap(window.key, window, error)
                except Exception:
                    logger.exception("取りこぼしの記録に失敗")
            self._deliver(window.key, order, None)
            return
        finally:
            with self._lock:
                self._pending -= 1
        self.usage.windows += 1
        self.usage.latencies.append(time.time() - started)
        self._deliver(window.key, order, (window, rows))

    def _deliver(self, key: str, order: int, payload: tuple | None) -> None:
        """**送った順**に呼び出し側へ渡す。

        順番が入れ替わると、話者の照合（時刻で候補を expire する）と画面の並びが狂う。
        2 系統を同時に投げているので、送り直しが入ると後ろの窓が先に返ることがある。
        """
        with self._lock:
            self._held[key][order] = payload
            ready = []
            while self._next[key] in self._held[key]:
                ready.append(self._held[key].pop(self._next[key]))
                self._next[key] += 1
        for item in ready:
            if item is None:
                continue
            window, rows = item
            try:
                self._on_rows(key, window, rows)
            except Exception:
                logger.exception("文字起こしの受け取りに失敗（%s %.1f 秒）", key, window.start_time)

    def _transcribe(self, window: AudioWindow) -> list[dict]:
        last: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                result = self.api.transcribe_audio(window.audio, window.sample_rate, self.work_dir)
                break
            except Exception as error:  # noqa: BLE001 — 送り直してから諦める
                last = error
                logger.warning("送信に失敗（%s 回目）: %s", attempt + 1, error)
                if attempt < self.config.retries:
                    self._sleep(self.config.retry_wait_sec)
        else:
            raise last if last is not None else RuntimeError("送信に失敗しました")
        usage = result.get("usage", {})
        self.usage.prompt_tokens += int(usage.get("promptTokenCount", 0) or 0)
        self.usage.output_tokens += int(usage.get("candidatesTokenCount", 0) or 0)
        return rows_in_meeting_time(window, result.get("utterances", []), result.get("words"))


def rows_in_meeting_time(window: AudioWindow, utterances: list[dict],
                         words: list[dict] | None = None, *, max_gap: float = 0.8) -> list[dict]:
    """返ってきたものを会議の時刻に戻し、**VAD の切れ目では必ず行を分ける**。

    語（`words`）が来ていればそちらを使う。無音を落として繋いで送っているので、外の側で
    まとめられた発話は**話者交代をまたいでいることがある**（2026-09-13 の通し確認で実際に、
    別の人の「なるほど。」が前の行に吸い込まれた）。区間の境目は元の音声では無音だった場所なので、
    そこで切れば手元の声紋が行ごとに当てられる。
    """
    if words:
        return _rows_from_words(window, words, max_gap=max_gap)
    rows: list[dict] = []
    for item in utterances:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = window.to_meeting_time(float(item.get("start", 0.0) or 0.0))
        end = window.to_meeting_time(float(item.get("end", 0.0) or 0.0))
        rows.append({"speaker": str(item.get("speaker") or ""), "text": text,
                     "start_time": round(start, 2), "end_time": round(max(end, start), 2)})
    return sorted(rows, key=lambda row: row["start_time"])


def _rows_from_words(window: AudioWindow, words: list[dict], *, max_gap: float = 0.8) -> list[dict]:
    """語を発話にまとめる。切るのは ①区間の境目 ②話者が替わったところ ③会議の時計で間が空いたところ。"""
    rows: list[dict] = []
    last_index: int | None = None
    for word in words:
        text = str(word.get("text", ""))
        if not text.strip():
            continue
        buffer_start = float(word.get("start", 0.0) or 0.0)
        index = window.span_index(buffer_start)
        start = window.to_meeting_time(buffer_start)
        end = window.to_meeting_time(float(word.get("end", 0.0) or 0.0))
        speaker = str(word.get("speaker") or "")
        same = (rows and index == last_index and rows[-1]["speaker"] == speaker
                and start - rows[-1]["end_time"] <= max_gap)
        if same:
            rows[-1]["text"] += text
            rows[-1]["end_time"] = round(max(end, rows[-1]["end_time"]), 2)
        else:
            rows.append({"speaker": speaker, "text": text,
                         "start_time": round(start, 2), "end_time": round(max(end, start), 2)})
        last_index = index
    for row in rows:
        row["text"] = row["text"].strip()
    return [row for row in rows if row["text"]]
