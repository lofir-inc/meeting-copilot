"""STT クライアントのユーティリティテスト（モデルロード不要）。"""

import sys
from types import SimpleNamespace

import numpy as np

from src.audio.vad import AudioChunk
from src.stt.whisper_client import SttConfig, WhisperClient


class TestResample:
    def test_downsample_48k_to_16k(self):
        # 48kHz 1秒 → 16kHz 1秒
        audio_48k = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
        audio_16k = WhisperClient._resample(audio_48k, 48000, 16000)

        assert audio_16k.shape[0] == 16000
        assert audio_16k.dtype == np.float32

    def test_resample_preserves_signal(self):
        # 単純な定数信号はリサンプリングしても同じ値
        audio = np.ones(48000, dtype=np.float32) * 0.5
        resampled = WhisperClient._resample(audio, 48000, 16000)

        np.testing.assert_array_almost_equal(resampled, 0.5, decimal=5)

    def test_upsample(self):
        audio_16k = np.random.randn(16000).astype(np.float32)
        audio_48k = WhisperClient._resample(audio_16k, 16000, 48000)

        assert audio_48k.shape[0] == 48000


def test_transcribe_passes_initial_prompt_and_disables_previous_text(monkeypatch):
    calls = []

    def fake_transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {"text": "十分なテキストです"}

    fake_module = SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)
    client = WhisperClient(SttConfig(initial_prompt="自分、固有名詞"))
    client._model_loaded = True
    chunk = AudioChunk("自分", np.ones(16000, dtype=np.float32), 0.0, 1.0, 16000)

    client.transcribe(chunk)

    assert calls[0]["initial_prompt"] == "自分、固有名詞"
    assert calls[0]["condition_on_previous_text"] is False


def _client_returning(monkeypatch, text: str, prompt: str | None) -> WhisperClient:
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=lambda audio, **kwargs: {"text": text}))
    client = WhisperClient(SttConfig(initial_prompt=prompt))
    client._model_loaded = True
    return client


def _chunk() -> AudioChunk:
    return AudioChunk("参加者B", np.ones(16000, dtype=np.float32), 0.0, 0.8, 16000)


class TestPromptEcho:
    """2026-09-11 本番: 息や物音の区間で「参加者A、参加者B」とヒントを読み上げた行が 3 割あった。"""

    PROMPT = "参加者A、参加者B"

    def test_names_only_output_is_dropped(self, monkeypatch):
        for text in ["参加者A、参加者B", "参加者B", "参加者B。", "参加者A 参加者B 参加者A"]:
            assert _client_returning(monkeypatch, text, self.PROMPT).transcribe(_chunk()) is None, text

    def test_sentence_that_mentions_a_name_is_kept(self, monkeypatch):
        segment = _client_returning(monkeypatch, "参加者Bさんこれ出しちゃえばいいよ", self.PROMPT).transcribe(_chunk())
        assert segment is not None
        assert segment.text == "参加者Bさんこれ出しちゃえばいいよ"

    def test_no_prompt_means_no_echo_filter(self, monkeypatch):
        assert _client_returning(monkeypatch, "参加者A、参加者B", None).transcribe(_chunk()) is not None


class TestLowConfidence:
    """2026-09-11 本番: 物音を「impressive」「лад振緻」などと起こした行が相手側に 100 行以上あった。"""

    def _client(self, monkeypatch, logprob: float, threshold):
        result = {"text": "impressive", "segments": [{"avg_logprob": logprob}]}
        monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=lambda audio, **kwargs: result))
        client = WhisperClient(SttConfig(min_avg_logprob=threshold))
        client._model_loaded = True
        return client

    def test_low_confidence_output_is_dropped(self, monkeypatch):
        assert self._client(monkeypatch, -1.4, -1.0).transcribe(_chunk()) is None

    def test_confident_output_is_kept(self, monkeypatch):
        assert self._client(monkeypatch, -0.5, -1.0).transcribe(_chunk()) is not None

    def test_filter_is_off_by_default(self, monkeypatch):
        """対面モードは SttConfig の既定のまま（判定しない）。"""
        assert self._client(monkeypatch, -3.0, None).transcribe(_chunk()) is not None
