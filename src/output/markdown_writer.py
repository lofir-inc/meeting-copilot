"""Markdown ファイル更新モジュール。

interview_summary.md (上書き) と interview_log.md (追記) を管理する。
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.llm.ollama_client import SummaryResponse
from src.llm.meeting_state import MeetingState

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    workspace_dir: str = "./workspace"
    save_audio_chunks: bool = False

    # 見出しラベル。既定は対面インタビュー用。
    # 会議モードは "会議" / "前回タスク 消化状況" を渡して読み替える。
    title: str = "インタビュー"
    items_label: str = "ヒアリング項目 進捗"

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir)

    @property
    def summary_path(self) -> Path:
        return self.workspace_path / "interview_summary.md"

    @property
    def log_path(self) -> Path:
        return self.workspace_path / "interview_log.md"

    @property
    def transcript_path(self) -> Path:
        return self.workspace_path / "transcripts.jsonl"

    @property
    def state_path(self) -> Path:
        return self.workspace_path / "state.json"

    @property
    def corrections_path(self) -> Path:
        return self.workspace_path / "corrections.jsonl"


class MarkdownWriter:
    """Markdown ファイルへの書き込みを管理する。"""

    def __init__(self, config: OutputConfig, interview_purpose: str = "") -> None:
        self.config = config
        self.interview_purpose = interview_purpose
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.config.workspace_path.mkdir(parents=True, exist_ok=True)

    def init_files(self, hearing_items: str) -> None:
        """インタビュー開始時に初期ファイルを作成する。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        summary_content = (
            f"# {self.config.title} ダッシュボード\n"
            f"> 最終更新: {now}\n\n"
            f"## 目的\n{self.interview_purpose or '（未設定）'}\n\n"
            f"## {self.config.items_label}\n{hearing_items}\n\n"
            f"## 全体要約\n（{self.config.title}開始待ち）\n\n"
            f"## AIからの提案\n（データ収集中…）\n"
        )
        self._write_file(self.config.summary_path, summary_content)

        log_content = (
            f"# {self.config.title} タイムラインログ\n"
            f"> 開始: {now}\n\n"
            f"---\n"
        )
        self._write_file(self.config.log_path, log_content)

        logger.info("Workspace files initialized: %s", self.config.workspace_path)

    def overwrite_summary(
        self,
        response: SummaryResponse,
        hearing_items_raw: str = "",
    ) -> None:
        """interview_summary.md を最新の要約で上書きする。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        progress_text = "\n".join(response.item_progress) if response.item_progress else "（判定中…）"
        suggestions_text = "\n".join(response.suggestions) if response.suggestions else "（提案なし）"

        content = (
            f"# {self.config.title} ダッシュボード\n"
            f"> 最終更新: {now}\n\n"
            f"## 目的\n{self.interview_purpose or '（未設定）'}\n\n"
            f"## {self.config.items_label}\n{progress_text}\n\n"
            f"## 全体要約\n{response.overall_summary or '（要約なし）'}\n\n"
            f"## AIからの提案\n{suggestions_text}\n"
        )

        self._write_file_locked(self.config.summary_path, content)
        logger.info("Summary updated at %s", now)

    def write_state(self, state: MeetingState) -> None:
        """ローリング状態を議事録 Markdown として上書きする。"""
        self._write_file_locked(
            self.config.summary_path,
            state.to_markdown(self.config.title + " ダッシュボード"),
        )

    def write_state_json(self, state: MeetingState) -> None:
        """ローリング状態を原子的に JSON ファイルへ書き込む。"""
        temporary_path = self.config.workspace_path / "state.json.tmp"
        temporary_path.write_text(state.to_json() + "\n", encoding="utf-8")
        os.replace(temporary_path, self.config.state_path)

    def append_log(self, chunk_summary: str, start_time: str = "", end_time: str = "") -> None:
        """interview_log.md の末尾にチャンクサマリーを追記する。"""
        now = datetime.now().strftime("%H:%M:%S")
        time_label = f"{start_time} - {end_time}" if start_time and end_time else now

        entry = f"\n---\n## {time_label}\n{chunk_summary}\n"

        self._append_file_locked(self.config.log_path, entry)
        logger.info("Log appended at %s", now)

    def append_transcript(self, segment_json: str) -> None:
        """生の文字起こしデータを JSONL ファイルに追記する。"""
        with open(self.config.transcript_path, "a", encoding="utf-8") as f:
            f.write(segment_json + "\n")

    def append_correction(self, record: dict) -> None:
        """発話単位の話者訂正を JSONL ファイルへ追記する。"""
        self._append_file_locked(self.config.corrections_path, json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Transcript correction appended: %s", self.config.corrections_path)

    @staticmethod
    def _write_file(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _write_file_locked(path: Path, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _append_file_locked(path: Path, content: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
