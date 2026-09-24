"""無音検知 (VAD) とチャンク切り出し。

チャンネルごとに独立した ChunkBuilder を保持し、
音声フレームを feed するたびに発話チャンクが確定したら yield する。
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_silero_model = None


@dataclass
class VadConfig:
    silence_threshold_db: float = -40
    silence_duration_sec: float = 1.5
    min_chunk_sec: float = 1.0
    max_chunk_sec: float = 30
    backend: str = "rms"
    speech_threshold: float = 0.5
    pre_roll_sec: float = 0.0


class VoiceDetector(Protocol):
    """音声フレームが発話を含むか判定する検出器。"""

    def is_voice(self, mono_frame: np.ndarray) -> bool:
        """フレームが発話を含むなら True を返す。"""

    def reset(self) -> None:
        """検出器の内部状態をリセットする。"""


class RmsDetector:
    """既存の RMS 閾値による音声検出器。"""

    def __init__(self, threshold_linear: float) -> None:
        self._threshold_linear = threshold_linear

    def is_voice(self, mono_frame: np.ndarray) -> bool:
        """既存と同じ RMS 比較で発話を判定する。"""
        rms = float(np.sqrt(np.mean(mono_frame ** 2)))
        return rms > self._threshold_linear

    def reset(self) -> None:
        """RMS 検出器にはリセット対象の状態がない。"""


class SileroDetector:
    """Silero VAD を 16 kHz・512 サンプル窓で呼び出す検出器。"""

    def __init__(self, sample_rate: int, threshold: float = 0.5) -> None:
        import torch
        from silero_vad import load_silero_vad

        global _silero_model
        torch.set_num_threads(1)
        if _silero_model is None:
            _silero_model = load_silero_vad()
            logger.info("Silero VAD model loaded")
        self._model = _silero_model
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._remainder = np.empty(0, dtype=np.float32)
        self._torch = torch

    def is_voice(self, mono_frame: np.ndarray) -> bool:
        """フレームを再サンプルし、含まれる窓の最大確率で判定する。"""
        frame = np.asarray(mono_frame, dtype=np.float32).reshape(-1)
        if self._sample_rate != 16000:
            frame = _resample_linear(frame, self._sample_rate, 16000)
        samples = np.concatenate((self._remainder, frame))
        window_count = samples.size // 512
        if window_count == 0:
            self._remainder = samples
            return False
        usable = samples[: window_count * 512]
        self._remainder = samples[window_count * 512 :]
        maximum = 0.0
        with self._torch.no_grad():
            for start in range(0, usable.size, 512):
                chunk = self._torch.from_numpy(usable[start : start + 512])
                probability = self._model(chunk, 16000).detach().item()
                maximum = max(maximum, float(probability))
        return maximum >= self._threshold

    def reset(self) -> None:
        """Silero の状態と端数サンプルを破棄する。"""
        self._remainder = np.empty(0, dtype=np.float32)
        self._model.reset_states()


def _resample_linear(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """線形補間で音声を再サンプルする。"""
    target_length = int(audio.size * target_rate / original_rate)
    if target_length <= 1 or audio.size <= 1:
        return audio.astype(np.float32)
    indices = np.linspace(0, audio.size - 1, target_length)
    return np.interp(indices, np.arange(audio.size), audio).astype(np.float32)


def make_detector(config: VadConfig, sample_rate: int) -> VoiceDetector:
    """設定に対応する音声検出器を作成する。"""
    if config.backend == "rms":
        return RmsDetector(10 ** (config.silence_threshold_db / 20))
    if config.backend == "silero":
        return SileroDetector(sample_rate, config.speech_threshold)
    raise ValueError(f"未知の VAD backend: {config.backend}")


@dataclass
class AudioChunk:
    """切り出し済みの音声チャンク。"""

    speaker: str          # "interviewer" / "guest"（対面）または話者名（会議）
    audio: np.ndarray     # float32 モノラル
    start_time: float     # セッション開始からの経過秒
    end_time: float
    sample_rate: int


class _State(enum.Enum):
    SILENT = "silent"
    SPEAKING = "speaking"


class ChunkBuilder:
    """1チャンネル分の VAD 状態マシン。

    時間管理は feed() に渡されたサンプル数から算出する（壁時計に依存しない）。
    """

    def __init__(
        self,
        speaker: str,
        config: VadConfig,
        sample_rate: int,
        session_start_offset: float = 0.0,
        detector: VoiceDetector | None = None,
    ) -> None:
        self.speaker = speaker
        self.config = config
        self.sample_rate = sample_rate
        self._total_frames: int = 0
        self._session_start_offset = session_start_offset
        self._state = _State.SILENT
        self._buffer: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_frames = 0
        self._chunk_start_time: float = 0.0
        self._silence_frames: int = 0
        self._detector = detector or make_detector(config, sample_rate)

    def skip(self, seconds: float) -> None:
        """feed せずに捨てた音声の分だけ時計を進める（録音ファイルの時刻と揃えるため）。"""
        self._total_frames += int(round(seconds * self.sample_rate))

    def _elapsed(self) -> float:
        """feed 済みサンプル数から経過秒を算出。"""
        return self._session_start_offset + self._total_frames / self.sample_rate

    def _buffer_duration(self) -> float:
        total_frames = sum(b.shape[0] for b in self._buffer)
        return total_frames / self.sample_rate

    def _remember_pre_roll(self, mono_data: np.ndarray) -> None:
        """無音中の直近フレームを設定秒数ぶん記録する。"""
        if self.config.pre_roll_sec <= 0:
            return
        self._pre_roll.append(mono_data)
        self._pre_roll_frames += mono_data.shape[0]
        limit = int(self.config.pre_roll_sec * self.sample_rate)
        while self._pre_roll and self._pre_roll_frames > limit:
            removed = self._pre_roll.popleft()
            self._pre_roll_frames -= removed.shape[0]

    def _flush_buffer(self) -> AudioChunk | None:
        if not self._buffer:
            return None
        audio = np.concatenate(self._buffer)
        duration = len(audio) / self.sample_rate
        self._buffer.clear()
        if duration < self.config.min_chunk_sec:
            return None
        return AudioChunk(
            speaker=self.speaker,
            audio=audio,
            start_time=self._chunk_start_time,
            end_time=self._chunk_start_time + duration,
            sample_rate=self.sample_rate,
        )

    def feed(self, mono_data: np.ndarray) -> list[AudioChunk]:
        """モノラル音声フレームを入力し、確定したチャンクをリストで返す。"""
        chunks: list[AudioChunk] = []
        is_voice = self._detector.is_voice(mono_data)
        frame_count = mono_data.shape[0]
        now = self._elapsed()
        if self._state == _State.SILENT:
            if is_voice:
                self._state = _State.SPEAKING
                pre_roll_duration = self._pre_roll_frames / self.sample_rate
                self._chunk_start_time = now - pre_roll_duration
                self._buffer.extend(self._pre_roll)
                self._pre_roll.clear()
                self._pre_roll_frames = 0
                self._buffer.append(mono_data)
                self._silence_frames = 0
            else:
                self._remember_pre_roll(mono_data)
        else:  # SPEAKING
            self._buffer.append(mono_data)
            if not is_voice:
                self._silence_frames += frame_count
                silence_sec = self._silence_frames / self.sample_rate
                if silence_sec >= self.config.silence_duration_sec:
                    chunk = self._flush_buffer()
                    if chunk:
                        chunks.append(chunk)
                    self._state = _State.SILENT
                    self._silence_frames = 0
            else:
                self._silence_frames = 0
            # 最大チャンク長に達したら強制切り出し
            if (
                self._state == _State.SPEAKING
                and self._buffer_duration() >= self.config.max_chunk_sec
            ):
                chunk = self._flush_buffer()
                if chunk:
                    chunks.append(chunk)
                self._chunk_start_time = self._elapsed()
        self._total_frames += frame_count
        return chunks
