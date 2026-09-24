"""ステレオ音声キャプチャモジュール。

sounddevice を使用してシステム入力からステレオ音声を常時キャプチャし、
Queue 経由で後段のパイプラインへ流す。
"""

from __future__ import annotations

import queue
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    channels: int = 2
    dtype: str = "float32"
    device: int | str | None = None
    chunk_duration_sec: float = 0.1


class AudioCapture:
    """ステレオ音声を常時キャプチャして Queue に積む。"""

    def __init__(self, config: AudioConfig, record_path: Path | None = None) -> None:
        self.config = config
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self._wav_file: wave.Wave_write | None = None

        if record_path:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav_file = wave.open(str(record_path), "wb")
            self._wav_file.setnchannels(config.channels)
            self._wav_file.setsampwidth(2)  # 16-bit
            self._wav_file.setframerate(config.sample_rate)
            logger.info("Recording to %s", record_path)

        blocksize = int(config.sample_rate * config.chunk_duration_sec)
        self._stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype=config.dtype,
            blocksize=blocksize,
            device=config.device,
            callback=self._callback,
        )

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        self.queue.put(indata.copy())
        if self._wav_file:
            # float32 → int16 に変換して書き込み
            pcm = (indata * 32767).astype(np.int16)
            self._wav_file.writeframes(pcm.tobytes())

    def start(self) -> None:
        logger.info(
            "Audio capture started (device=%s, rate=%d, blocksize=%.1fs)",
            self.config.device,
            self.config.sample_rate,
            self.config.chunk_duration_sec,
        )
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()
        if self._wav_file:
            self._wav_file.close()
            logger.info("Recording saved")
        logger.info("Audio capture stopped")

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
