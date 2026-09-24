

class TestPerEngineWindow:
    """2026-09-18 運用者 指摘「エンジンごとに適切なタイミングになっているか」。

    30 秒は Whisper の都合で決めた値だった。Deepgram は往復 1.5 秒（実会議 98 窓の中央）
    なので、30 秒も溜める理由がない。
    """

    def test_Deepgramは短い窓にする(self):
        from src.stt.live_batch import defaults_for

        assert defaults_for("deepgram")["window_sec"] == 8.0

    def test_Geminiは長いまま(self):
        """往復 12.6 秒で効き目が薄いうえ、レート制限 10 リクエスト/分に当たる。"""
        from src.stt.live_batch import defaults_for

        assert defaults_for("gemini")["window_sec"] == 30.0

    def test_知らないエンジンには口を出さない(self):
        from src.stt.live_batch import defaults_for

        assert defaults_for("なにか") == {}

    def test_設定に書いた値が勝つ(self):
        """auto をやめて手で決められること（運用者「柔軟にコントロールしてくれるといい」）。"""
        from src.stt.live_batch import LiveBatchConfig, defaults_for

        config = LiveBatchConfig(**{**defaults_for("deepgram"), "window_sec": 15.0})

        assert config.window_sec == 15.0
        assert config.first_window_sec == 4.0      # 書いていないほうは既定のまま
