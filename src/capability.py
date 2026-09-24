"""この Mac で何ができるかを見る（できないものを画面に出さないため）。

会議を始めてから「手元の文字起こしが動きません」と分かるのがいちばん困る。
起動の前に分かることは先に見ておく。

見るだけ。重い読み込み（Whisper のモデルなど）はしない。1 秒以内で返す。
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

MIN_MEMORY_GB_LOCAL = 24
"""会議中に手元で回す（Whisper ＋ Ollama）ために要るメモリの目安。

実測（2026-09-11・MacBook Pro M4 Max / 128GB）: 手元だけで回すと**文字起こしが 1〜2 分遅れ**、
MacWhisper と食い合って GPU リセットが 2 回。16GB 機ではまず成立しない。
24GB を境にしているのは「遅れても動く」の下限（運用者 と相談のうえ決める余地あり）。
"""


@dataclass
class Capability:
    """この Mac でできること。"""

    apple_silicon: bool = False
    memory_gb: int = 0
    ollama: bool = False
    ollama_models: list[str] = field(default_factory=list)
    claude_cli: bool = False
    gemini_key: bool = False
    deepgram_key: bool = False

    @property
    def local_stt(self) -> bool:
        """会議中の文字起こしを手元で回せるか（mlx-whisper は Apple Silicon 前提）。"""
        return self.apple_silicon

    @property
    def local_llm(self) -> bool:
        """会議中の要約を手元で回せるか。"""
        return self.ollama and bool(self.ollama_models)

    @property
    def offline(self) -> bool:
        """ネットが無くても会議を通せるか（ハイスペック機だけ）。"""
        return self.local_stt and self.local_llm and self.memory_gb >= MIN_MEMORY_GB_LOCAL

    def as_dict(self) -> dict:
        return {"apple_silicon": self.apple_silicon, "memory_gb": self.memory_gb,
                "ollama": self.ollama, "ollama_models": self.ollama_models,
                "claude_cli": self.claude_cli, "gemini_key": self.gemini_key,
                "deepgram_key": self.deepgram_key,
                "local_stt": self.local_stt, "local_llm": self.local_llm, "offline": self.offline}

    def why_not_offline(self) -> list[str]:
        """オフラインで回せない理由（画面にそのまま出す）。"""
        reasons = []
        if not self.apple_silicon:
            reasons.append("Apple Silicon ではありません（手元の文字起こしに mlx-whisper を使います）")
        if not self.ollama:
            reasons.append("Ollama が入っていません")
        elif not self.ollama_models:
            reasons.append("Ollama にモデルが降りていません（`ollama pull gemma4` など）")
        if self.memory_gb and self.memory_gb < MIN_MEMORY_GB_LOCAL:
            reasons.append(f"メモリが {self.memory_gb}GB です"
                           f"（会議中に文字起こしと要約を同時に回すには {MIN_MEMORY_GB_LOCAL}GB ほど要ります）")
        return reasons


def _memory_gb() -> int:
    """実装メモリ（GB）。取れなければ 0（判定に使わない）。"""
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
        return round(int(result.stdout.strip()) / 1024 ** 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _ollama_models(timeout: float = 3.0) -> list[str]:
    """降りているモデルの名前。Ollama が動いていなければ空（立ち上げには行かない）。"""
    if shutil.which("ollama") is None:
        return []
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    names = []
    for line in result.stdout.splitlines()[1:]:      # 1 行目は見出し
        name = line.split()[0] if line.split() else ""
        if name:
            names.append(name)
    return names


def look(settings: dict | None = None) -> Capability:
    """いまの Mac を見る。1 秒以内で返す（会議の前に毎回呼べるように）。"""
    meeting = (settings or {}).get("meeting") or {}
    key_file = str(((meeting.get("external_stt") or {}).get("key_file") or "")).strip()
    return Capability(
        apple_silicon=platform.system() == "Darwin" and platform.machine() == "arm64",
        memory_gb=_memory_gb(),
        ollama=shutil.which("ollama") is not None,
        ollama_models=_ollama_models(),
        claude_cli=shutil.which("claude") is not None,
        gemini_key=bool(key_file) and Path(key_file).expanduser().is_file(),
        deepgram_key=_has_key((meeting.get("external_stt") or {}).get("deepgram_key_file", "")),
    )


def _has_key(told: str) -> bool:
    """キーの置き場所が設定されていて、実際にファイルがあるか。中身は見ない。"""
    told = str(told or "").strip()
    return bool(told) and Path(told).expanduser().is_file()


def main() -> int:                                   # pragma: no cover - 手で確かめる用
    import sys

    import yaml

    repo = Path(__file__).resolve().parent.parent
    path = repo / "config" / "settings.yaml"
    settings = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    found = look(settings)
    print(json.dumps(found.as_dict(), ensure_ascii=False, indent=2))
    if not found.offline:
        print("\nオフラインでは回せません:")
        for reason in found.why_not_offline():
            print("  -", reason)
    return 0


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
