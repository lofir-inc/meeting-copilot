"""Deepgram へ送る口と、その関門（学習に使われない口かを機械で確かめる）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.stt.deepgram_transcribe import DeepgramApi, DeepgramConfig


class TestGate:
    """関門: 送ったものが学習に使われず、保存もされないことを機械で確かめる。

    2026-09-18 に作り直した。opt out は**リクエストごとのパラメータ**で、プロジェクト設定を
    PATCH しても **200 が返るのに変わらない**（黙って無視される）。だから「設定したか」では守れない。
    """

    def test_毎回optoutを付けて送る(self, tmp_path, monkeypatch):
        api = DeepgramApi("key")
        seen = {}
        monkeypatch.setattr(api, "_request_url", lambda: None, raising=False)

        def fake_open(request, timeout=0):
            seen["url"] = request.full_url
            raise RuntimeError("ここまで見れば十分")

        monkeypatch.setattr("urllib.request.urlopen", fake_open)
        try:
            api._post(b"RIFF")
        except Exception:
            pass

        assert "mip_opt_out=true" in seen["url"]

    def test_付けない設定なら通さない(self):
        api = DeepgramApi("key", DeepgramConfig(mip_opt_out=False))

        found = api.data_use()

        assert found["ok"] is False and "学習に使われます" in found["why"]

    def test_受け付けられれば通す(self, monkeypatch):
        api = DeepgramApi("key")
        monkeypatch.setattr(api, "_post", lambda body: {"metadata": {"request_id": "abc-123"}})

        found = api.data_use()

        assert found["ok"] is True and found["request_id"] == "abc-123"

    def test_試せなければ通さない(self, monkeypatch):
        """fail-closed（落ちている・権限が無い・オフライン）。"""
        api = DeepgramApi("key")

        def boom(body):
            raise RuntimeError("401")

        monkeypatch.setattr(api, "_post", boom)

        assert api.data_use()["ok"] is False

    def test_キーが無ければ作れない(self):
        with pytest.raises(ValueError):
            DeepgramApi("")


class TestParse:
    """会議中の経路が読める形に直す（語つき）。"""

    PAYLOAD = {
        "metadata": {"duration": 12.5},
        "results": {"channels": [{"alternatives": [{
            "transcript": "こんにちは。よろしくお願いします。",
            "words": [{"word": "こんにちは", "punctuated_word": "こんにちは。", "start": 0.1, "end": 1.2},
                      {"word": "よろしく", "punctuated_word": "よろしく", "start": 2.0, "end": 2.8}],
        }]}]},
    }

    def test_語と発話を取り出す(self):
        result = DeepgramApi("key")._parse(self.PAYLOAD)

        assert result["words"][0] == {"text": "こんにちは。", "start": 0.1, "end": 1.2, "speaker": ""}
        assert result["utterances"][0]["text"].startswith("こんにちは")
        assert result["usage"]["audioSeconds"] == 12.5

    def test_空の返事でも落ちない(self):
        assert DeepgramApi("key")._parse({}) == {"utterances": [], "words": [],
                                                 "usage": {"audioSeconds": 0.0}}


def test_会議中の刻みはwavにして送る(tmp_path, monkeypatch):
    """Gemini の TranscribeApi と同じ形（差し替えるだけで経路が動く）。"""
    api = DeepgramApi("key", DeepgramConfig())
    sent = {}

    def fake_post(body):
        sent["bytes"] = len(body)
        return TestParse.PAYLOAD

    monkeypatch.setattr(api, "_post", fake_post)
    audio = np.zeros(16000, dtype=np.float32)

    result = api.transcribe_audio(audio, 16000, tmp_path)

    assert sent["bytes"] > 32000 and result["words"]
    assert not list(tmp_path.glob("*.wav"))          # 送ったら消す
