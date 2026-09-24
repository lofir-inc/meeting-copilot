"""Agent CLI を安全な引数配列で呼び出すクライアント。"""

from __future__ import annotations

import json
import shutil
import subprocess


class AgentError(RuntimeError):
    """Agent CLI が失敗した場合の例外。"""


class AgentClient:
    """Agent の机作成だけを提供する薄いクライアント。"""

    def __init__(self, binary: str = "agent", timeout: int = 30) -> None:
        self.binary = binary
        self.timeout = timeout

    def available(self) -> bool:
        """Agent CLI が PATH 上にあるかを返す。"""
        return shutil.which(self.binary) is not None

    def create_terminal(self, repo: str, title: str, command: str, focus: bool = False) -> dict:
        """指定リポジトリを対象とする Agent の机を作成する。"""
        argv = [self.binary, "terminal", "create", "--worktree", f"path:{repo}", "--title", title, "--command", command, "--json"]
        if focus:
            argv.append("--focus")
        result = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=self.timeout)
        if result.returncode != 0:
            raise AgentError(result.stderr.strip() or f"agent exited with {result.returncode}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AgentError("agent の JSON 応答を解釈できません") from exc
