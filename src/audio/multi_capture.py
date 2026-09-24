"""複数入力デバイスの同時キャプチャ。

会議（ループバック）モード専用。対面モードの `capture.py` には触れない。

なぜ2本開くのか
  ループバック（BlackHole）には**自分の声が入らない**。会議アプリは自分の声を返さないため
  （Zoom・Google Meet・Teams いずれも同じ）。
  1本にまとめると司会＝自分さんの発言だけが丸ごと消える。ここが一番踏みやすい罠。
  ∴ 自分のマイクとループバックを別ストリームで開き、混ぜずにラベルを付ける。

2デバイスは別クロックなのでドリフトする
  音を混ぜず、時刻ラベルにしか使わないので影響しない。ただし**開始時刻は揃える**必要が
  あるため、各ストリームの初回コールバックの壁時計時刻を記録し、共通の t=0 からの
  オフセットとして後段（ChunkBuilder）へ渡す。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.audio.sck_capture import SckConfig, SckSystemAudio

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """1入力ソースの設定。"""

    key: str            # "self" / "remote" — ストリームの識別子
    device: int | None  # sounddevice のデバイス番号。ScreenCaptureKit では None。
    device_name: str    # 表示・ログ用の実名
    channels: int       # そのデバイスの入力チャンネル数
    speaker: str | None = None
    """話者ラベルを固定する場合はここに名前を入れる。

    None なら「チャンクごとに話者判定が要る」ことを意味する（ループバック側）。
    """
    backend: str = "sounddevice"
    """取得バックエンド。`sounddevice` または `screencapturekit` を指定する。"""


@dataclass
class MultiAudioConfig:
    sample_rate: int = 48000
    dtype: str = "float32"
    chunk_duration_sec: float = 0.1


class MultiCapture:
    """複数の入力デバイスを同時に開き、ソースごとに別 Queue へ積む。

    キューに入るのは**モノラルの float32 配列**。多チャンネル入力はチャンネル平均で
    モノラルへ落とす（ループバックのステレオは左右とも同じ会議音声のミックスであり、
    分離の手掛かりにはならないため）。
    """

    def __init__(
        self,
        sources: list[SourceConfig],
        config: MultiAudioConfig,
        record_dir: Path | None = None,
        sck: dict | None = None,
    ) -> None:
        if not sources:
            raise ValueError("sources が空です。最低1つの入力ソースが必要です。")

        keys = [s.key for s in sources]
        if len(keys) != len(set(keys)):
            raise ValueError(f"ソースの key が重複しています: {keys}")

        self.config = config
        self._sck_overrides = sck or {}
        self.sources: dict[str, SourceConfig] = {s.key: s for s in sources}
        self.queues: dict[str, queue.Queue[np.ndarray]] = {s.key: queue.Queue() for s in sources}

        self._streams: dict[str, sd.InputStream] = {}
        self._sck_streams: dict[str, SckSystemAudio] = {}
        self._wav_files: dict[str, wave.Wave_write] = {}
        self._first_frame_at: dict[str, float] = {}
        self._total_samples: dict[str, int] = {s.key: 0 for s in sources}
        self._taken: dict[str, int] = {s.key: 0 for s in sources}
        """キューから取り出したサンプル数（セルフチェック・登録で捨てた分も含む）。"""
        self._lock = threading.Lock()
        self._first_frame_event = threading.Event()
        self._started = False

        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)

        blocksize = int(config.sample_rate * config.chunk_duration_sec)

        for source in sources:
            if record_dir is not None:
                wav_path = record_dir / f"recording_{source.key}.wav"
                if wav_path.exists() and wav_path.stat().st_size > 0:
                    # 同じセッションで立ち上げ直すと、前の録音を頭から書き潰していた（2026-09-11 本番で踏みかけた）
                    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(wav_path.stat().st_mtime))
                    kept = wav_path.with_name(f"recording_{source.key}.{stamp}.wav")
                    wav_path.rename(kept)
                    logger.warning("前回の録音を %s に退避しました", kept.name)
                wav = wave.open(str(wav_path), "wb")
                wav.setnchannels(1)      # モノラルに落として保存する
                wav.setsampwidth(2)      # 16-bit
                wav.setframerate(config.sample_rate)
                self._wav_files[source.key] = wav
                logger.info("Recording [%s] to %s", source.key, wav_path)

            if source.backend == "sounddevice":
                self._streams[source.key] = sd.InputStream(
                    samplerate=config.sample_rate,
                    channels=source.channels,
                    dtype=config.dtype,
                    blocksize=blocksize,
                    device=source.device,
                    callback=self._make_callback(source.key),
                )
            elif source.backend == "screencapturekit":
                sck_values = {
                    "sample_rate": config.sample_rate,
                    "channels": 1,
                    "frame_sec": config.chunk_duration_sec,
                }
                sck_values.update(self._sck_overrides)
                self._sck_streams[source.key] = SckSystemAudio(
                    SckConfig(**sck_values),
                    on_frame=lambda frame, key=source.key: self._ingest(key, frame),
                )
            else:
                raise ValueError(f"未対応の audio backend です: {source.backend}")

    def _make_callback(self, key: str):
        """ソースごとのコールバックを生成する（key を閉じ込める）。"""

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                logger.warning("Audio callback status [%s]: %s", key, status)

            mono = to_mono(indata)

            self._ingest(key, mono)

        return _callback

    def _ingest(self, key: str, mono: np.ndarray) -> None:
        """モノラルフレームを記録し、キューと必要なら WAV へ渡す。"""
        frame = np.asarray(mono, dtype=np.float32).reshape(-1).copy()
        with self._lock:
            if key not in self._first_frame_at:
                self._first_frame_at[key] = time.time()
                logger.debug("First frame on [%s]", key)
                if len(self._first_frame_at) == len(self.sources):
                    self._first_frame_event.set()
            self._total_samples[key] += len(frame)
        self.queues[key].put(frame)
        wav = self._wav_files.get(key)
        if wav is not None:
            wav.writeframes((np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    def start(self, first_frame_timeout: float = 5.0) -> None:
        for key, stream in self._streams.items():
            source = self.sources[key]
            logger.info(
                "Audio capture started [%s]: device=[%d] %s (ch=%d, rate=%d)",
                key, source.device, source.device_name, source.channels, self.config.sample_rate,
            )
            stream.start()
        for key, stream in self._sck_streams.items():
            logger.info("Audio capture started [%s]: ScreenCaptureKit (rate=%d)", key, self.config.sample_rate)
            stream.start(timeout=first_frame_timeout)
        self._started = True

    def wait_for_first_frames(self, timeout: float = 5.0) -> bool:
        """全ストリームが最初のフレームを届けるまで待つ。

        時刻合わせのオフセットを確定させるために使う。
        timeout 内に揃わなければ False（届いていないソースがある＝要警告）。
        """
        return self._first_frame_event.wait(timeout)

    def start_offsets(self) -> dict[str, float]:
        """共通の t=0 から見た、各ストリームの開始オフセット（秒）を返す。

        最も早く始まったストリームを 0 とする。まだフレームが来ていないソースは
        含まれない（呼び出し側で欠落を検出できるようにする）。
        """
        with self._lock:
            firsts = dict(self._first_frame_at)
        if not firsts:
            return {}
        base = min(firsts.values())
        return {key: at - base for key, at in firsts.items()}

    def silent_sources(self) -> list[str]:
        """1フレームも届いていないソースの key を返す。"""
        with self._lock:
            arrived = set(self._first_frame_at)
        return [key for key in self.sources if key not in arrived]

    def drain(self, key: str) -> list[np.ndarray]:
        """指定ソースのキューに溜まっているモノラルフレームを全部取り出す。"""
        frames: list[np.ndarray] = []
        q = self.queues[key]
        while True:
            try:
                frames.append(q.get_nowait())
            except queue.Empty:
                break
        self._taken[key] += sum(len(frame) for frame in frames)
        return frames

    def trim_backlog(self, key: str, keep_seconds: float) -> float:
        """溜まりのうち末尾 keep_seconds ぶんを残し、古い分だけ捨てる。捨てた秒数を返す。

        本編前の挨拶・雑談を文字起こしに残すため（2026-09-12 運用者 の運用フロー: 参加して
        挨拶しながら相手側のレベルを測り、その挨拶の行で話者に名前を付ける）。
        全部捨てていた頃は、挨拶が1行も出ないので名前を付ける対象が無かった。
        """
        frames = self.drain(key)
        target = int(round(keep_seconds * self.config.sample_rate))
        kept: list[np.ndarray] = []
        kept_samples = 0
        for frame in reversed(frames):
            if kept_samples >= target:
                break
            kept.append(frame)
            kept_samples += len(frame)
        kept.reverse()
        for frame in kept:
            self.queues[key].put(frame)
        # 戻した分は「まだ取り出していない」＝録音との時刻合わせで二重に数えない
        self._taken[key] -= kept_samples
        dropped = sum(len(frame) for frame in frames) - kept_samples
        return dropped / self.config.sample_rate

    def taken_seconds(self, key: str) -> float:
        """これまでにキューから取り出した音声の長さ（秒）。録音ファイルの先頭からの位置と一致する。"""
        return self._taken[key] / self.config.sample_rate

    def record_from(self, key: str, seconds: float, discard_backlog: bool = True) -> np.ndarray:
        """指定秒数ぶんのフレームをキューから取り出して結合する。

        既定でキューの溜まりを捨ててから「これから」の音を取る。キャプチャは登録より前に
        始まっているので、捨てないと「名前を入力している間に溜まった古い音」が登録音声になる
        （bb レビュー 2026-09-09）。
        """
        if not self._started:
            raise RuntimeError("キャプチャが開始されていません。")
        if discard_backlog:
            self.drain(key)
        target = int(round(seconds * self.config.sample_rate))
        frames: list[np.ndarray] = []
        samples = 0
        while samples < target:
            try:
                frame = self.queues[key].get(timeout=max(0.1, seconds))
            except queue.Empty:
                break
            frames.append(frame)
            samples += len(frame)
        self._taken[key] += samples
        return np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)

    def drift(self) -> dict[str, float]:
        """初回フレームからの壁時計とサンプル時計の差をソースごとに返す。"""
        now = time.time()
        with self._lock:
            return {
                key: (now - started) - self._total_samples[key] / self.config.sample_rate
                for key, started in self._first_frame_at.items()
            }

    def sck_stats(self) -> dict:
        """ScreenCaptureKit ソースの統計を返す。"""
        return {key: stream.stats() for key, stream in self._sck_streams.items()}

    def stop(self) -> None:
        for key, stream in self._streams.items():
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("ストリームの停止に失敗 [%s]", key)
        for key, stream in self._sck_streams.items():
            try:
                stream.stop()
            except Exception:
                logger.exception("ScreenCaptureKit ストリームの停止に失敗 [%s]", key)
        for key, wav in self._wav_files.items():
            try:
                wav.close()
                logger.info("Recording saved [%s]", key)
            except Exception:
                logger.exception("録音ファイルのクローズに失敗 [%s]", key)
        self._started = False
        logger.info("Audio capture stopped (%d sources)", len(self._streams))

    def __enter__(self) -> "MultiCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def to_mono(data: np.ndarray) -> np.ndarray:
    """多チャンネル配列をモノラル float32 に落とす（チャンネル平均）。"""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        return array.copy()
    if array.shape[1] == 1:
        return array[:, 0].copy()
    return array.mean(axis=1).astype(np.float32)
