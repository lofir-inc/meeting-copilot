"""この Mac で何ができるか（できない選び方を画面に出さないため）。"""

from __future__ import annotations

import pytest

from src import capability


def make(**kwargs) -> capability.Capability:
    base = {"apple_silicon": True, "memory_gb": 36, "ollama": True,
            "ollama_models": ["gemma4:latest"], "claude_cli": True, "gemini_key": True,
            "deepgram_key": True}
    return capability.Capability(**{**base, **kwargs})


class TestWhatItCanDo:
    def test_そろっていればオフラインで回せる(self):
        assert make().offline is True

    def test_IntelMacでは手元の文字起こしができない(self):
        """mlx-whisper は Apple Silicon 前提。"""
        found = make(apple_silicon=False)

        assert found.local_stt is False and found.offline is False
        assert any("Apple Silicon" in reason for reason in found.why_not_offline())

    def test_モデルが降りていなければ手元の要約ができない(self):
        found = make(ollama_models=[])

        assert found.local_llm is False
        assert any("モデル" in reason for reason in found.why_not_offline())

    def test_メモリが足りなければオフラインにしない(self):
        """実測 2026-09-11（M4 Max / 128GB）でも文字起こしが 1〜2 分遅れた。16GB では成立しない。"""
        found = make(memory_gb=16)

        assert found.local_stt is True and found.offline is False
        assert any("メモリ" in reason for reason in found.why_not_offline())

    def test_見るだけで重い読み込みをしない(self):
        """会議の前に毎回呼ぶので、1 秒以内に返ること。"""
        import time

        start = time.time()
        capability.look({})

        assert time.time() - start < 3.0


class TestPresets:
    def test_できない動かし方は理由つきで返す(self):
        from src import settings_edit

        presets = {one["name"]: one for one in settings_edit.presets_for(make(ollama=False, ollama_models=[]))}

        assert presets["quality"]["usable"] is True          # 外で回すぶんには困らない
        assert presets["secret"]["usable"] is False
        assert "手元の LLM" in "".join(presets["secret"]["missing"])

    def test_できない動かし方は当てない(self, tmp_path):
        """会議が始まってから「動きません」と分かるのを避ける。"""
        from src import settings_edit

        path = tmp_path / "settings.yaml"
        path.write_text("stt:\n  engine: gemini\n", encoding="utf-8")

        with pytest.raises(ValueError, match="選べません"):
            settings_edit.apply_preset(path, "secret", make(apple_silicon=False))

    def test_当てると処理ごとのエンジンがまとめて変わる(self, tmp_path):
        from src import settings_edit

        path = tmp_path / "settings.yaml"
        path.write_text("stt:\n  engine: gemini\n\nllm:\n  engine: gemini\n  final_pass: none\n\n"
                        "minutes:\n  engine: auto\n\nmeeting:\n  verify_engine: auto\n"
                        "  external_stt:\n    enabled: true\n", encoding="utf-8")

        result = settings_edit.apply_preset(path, "secret", make())

        assert result["label"] == "機密優先"
        values = {field["key"]: field["value"] for field in settings_edit.read_fields(path)}
        assert values["stt.engine"] == "local" and values["llm.engine"] == "local"
        assert values["meeting.external_stt.enabled"] is False
        assert settings_edit.current_preset(path) == "secret"


class TestPresetsReachTheMeeting:
    """プリセットが入れる値を、会議側が「外で回す」と読めるか。

    2026-09-18 の穴: 会議側が `stt.engine == "gemini"` だけを見ていたため、
    `deepgram` を入れるプリセットを当てると**黙って手元へ落ちていた**。
    """

    def test_外へ出すプリセットの値を会議側が外だと読む(self):
        from src import settings_edit
        from src.meeting_orchestrator import EXTERNAL_STT_ENGINES

        for preset in settings_edit.PRESETS:
            engine = preset["values"].get("stt.engine")
            goes_out = preset["values"].get("meeting.external_stt.enabled") is True
            assert (engine in EXTERNAL_STT_ENGINES) is goes_out, (
                f"{preset['name']}: stt.engine={engine} と external_stt.enabled={goes_out} が食い違う")

    def test_手元だけのプリセットは外だと読まれない(self):
        from src import settings_edit
        from src.meeting_orchestrator import EXTERNAL_STT_ENGINES

        secret = {one["name"]: one for one in settings_edit.PRESETS}["secret"]

        assert secret["values"]["stt.engine"] not in EXTERNAL_STT_ENGINES
