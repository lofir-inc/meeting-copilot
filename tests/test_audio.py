"""音声パイプライン（splitter / VAD）の単体テスト。"""

import numpy as np
import pytest

from src.audio.splitter import split_channels
from src.audio.vad import AudioChunk, ChunkBuilder, VadConfig


class TestSplitChannels:
    def test_split_returns_two_mono_arrays(self):
        stereo = np.random.randn(4800, 2).astype(np.float32)
        left, right = split_channels(stereo)
        assert left.shape == (4800,)
        assert right.shape == (4800,)

    def test_split_preserves_values(self):
        stereo = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
        left, right = split_channels(stereo)
        np.testing.assert_array_almost_equal(left, [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(right, [0.9, 0.8, 0.7])

    def test_split_returns_copies(self):
        stereo = np.array([[1.0, 2.0]], dtype=np.float32)
        left, right = split_channels(stereo)
        left[0] = 999.0
        assert stereo[0, 0] == 1.0  # 元データは変わらない


class TestChunkBuilder:
    @pytest.fixture
    def config(self):
        return VadConfig(
            silence_threshold_db=-40,
            silence_duration_sec=0.3,
            min_chunk_sec=0.2,
            max_chunk_sec=5.0,
        )

    def _make_silence(self, duration_sec: float, sample_rate: int = 48000) -> np.ndarray:
        return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)

    def _make_voice(self, duration_sec: float, sample_rate: int = 48000, amplitude: float = 0.1) -> np.ndarray:
        t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), dtype=np.float32)
        return amplitude * np.sin(2 * np.pi * 440 * t)

    def test_silence_produces_no_chunks(self, config):
        builder = ChunkBuilder("interviewer", config, 48000)
        silence = self._make_silence(1.0)
        # feed in small blocks
        block_size = 4800
        all_chunks = []
        for i in range(0, len(silence), block_size):
            all_chunks.extend(builder.feed(silence[i:i + block_size]))
        assert len(all_chunks) == 0

    def test_voice_then_silence_produces_chunk(self, config):
        builder = ChunkBuilder("guest", config, 48000)
        voice = self._make_voice(0.5)
        silence = self._make_silence(0.5)

        chunks = []
        block_size = 4800
        audio = np.concatenate([voice, silence])
        for i in range(0, len(audio), block_size):
            chunks.extend(builder.feed(audio[i:i + block_size]))

        assert len(chunks) == 1
        assert chunks[0].speaker == "guest"
        assert chunks[0].audio.shape[0] > 0

    def test_short_voice_discarded(self, config):
        config.min_chunk_sec = 1.0  # 最小1秒
        builder = ChunkBuilder("interviewer", config, 48000)

        short_voice = self._make_voice(0.1)
        silence = self._make_silence(0.5)
        audio = np.concatenate([short_voice, silence])

        chunks = []
        block_size = 4800
        for i in range(0, len(audio), block_size):
            chunks.extend(builder.feed(audio[i:i + block_size]))

        assert len(chunks) == 0  # min_chunk_sec 未満なので破棄

    def test_max_chunk_forces_split(self, config):
        config.max_chunk_sec = 0.5
        builder = ChunkBuilder("guest", config, 48000)

        long_voice = self._make_voice(1.5)
        chunks = []
        block_size = 4800
        for i in range(0, len(long_voice), block_size):
            chunks.extend(builder.feed(long_voice[i:i + block_size]))

        assert len(chunks) >= 2  # 0.5秒ごとに強制切り出し
