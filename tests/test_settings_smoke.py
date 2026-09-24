"""config/settings.yaml が、各モードの設定 dataclass にそのまま流し込めることの煙テスト。

2026-09-09: settings に `llm.final_pass` を足したら、対面モードの Orchestrator と replay の
`LlmConfig(**config["llm"])` が TypeError で落ちた（単体テストは settings.yaml を読まないので気づけなかった）。
このテストは「settings.yaml の全セクションが受け側の dataclass に入る」ことだけを見る。
"""

from pathlib import Path

import yaml

from src.audio.capture import AudioConfig
from src.audio.vad import VadConfig
from src.dispatch import DispatchConfig
from src.llm.ollama_client import LlmConfig
from src.llm.state_updater import StateUpdaterConfig
from src.meeting_orchestrator import MeetingConfig
from src.task_hub import TaskHubConfig
from src.output.markdown_writer import OutputConfig
from src.stt.whisper_client import SttConfig
from src.ui.server import UiConfig

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
# 配布版には settings.yaml が無い（利用者が example から作る）。あるほうを見る。
SETTINGS = next(path for path in (CONFIG_DIR / "settings.yaml", CONFIG_DIR / "settings.example.yaml")
                if path.exists())


def _config() -> dict:
    return yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))


def test_settings_sections_fit_their_dataclasses():
    config = _config()
    audio = dict(config["audio"])
    audio.pop("device_candidates", None)   # main.py が取り除いてから渡す
    AudioConfig(**audio)
    VadConfig(**config["vad"])
    SttConfig(**config["stt"])
    LlmConfig(**config["llm"])             # final_pass を含んだまま受けられること（対面モードと replay の経路）
    OutputConfig(**config["output"])
    UiConfig(**config["ui"])
    TaskHubConfig(**config["task_hub"])
    DispatchConfig(**config["dispatch"])
    meeting = MeetingConfig(**config["meeting"])
    StateUpdaterConfig(**meeting.state)
    vad_values = dict(config["vad"])
    vad_values.update(meeting.vad)
    VadConfig(**vad_values)


def test_llm_final_pass_defaults_to_none():
    assert LlmConfig().final_pass == "none"
    assert LlmConfig(**_config()["llm"]).final_pass in {"none", "claude_cli"}
