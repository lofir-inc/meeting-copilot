"""タスク管理 の既存入口を会議モードから利用するための橋渡し。"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

import httpx

from src.llm.meeting_state import Item

logger = logging.getLogger(__name__)


class TaskHubUnavailable(RuntimeError):
    """タスク管理 の共有モジュールまたはクライアント情報を利用できない場合の例外。"""


@dataclass
class TaskHubConfig:
    """タスク管理 連携の設定。"""

    client_name: str = "自社"
    shared_dir: str = "~/.claude/skills/_shared"
    chat_notify_url: str = "https://example.invalid/webhook/chat-notify"
    slack: bool = True
    notion_tasks: bool = True
    people_master_db: str = ""
    """People Master DB（話者ラベル → 人）。哲学カードの行き先は 連携先と同じ 2 段で決める。"""

    personas: dict = field(default_factory=dict)
    """会議の話者名 → Notion のペルソナ名（哲学カードの行き先。自分 → 自社自分）。

    resolve_persona は部分一致なので付けなくても当たるが、**別人に当たっても黙って登録される**。
    ここで明示し、当たった名前は登録のたびに画面へ出す（運用者 指摘 2026-09-17）。
    """

    dictionary_sync: bool = False
    """タスク管理 のクライアント辞書とつなぐか（読む: 会議の終わりにキャッシュ／書く: 選んだ候補を Status=候補 で）。
    中身は `src/text/task_hub_dictionary.py`。"""


@dataclass
class ClientRef:
    """Client Master から得る タスク管理 のクライアント参照。"""

    client_id: str
    client_name: str
    task_db_id: str
    minutes_db_id: str


class TaskHubBridge:
    """タスク管理 のタスク、通知、クライアント解決を薄く包む。"""

    def __init__(self, cfg: TaskHubConfig) -> None:
        self.cfg = cfg
        self._client: ClientRef | None = None
        shared_dir = Path(cfg.shared_dir).expanduser()
        if shared_dir.is_dir() and str(shared_dir) not in sys.path:
            sys.path.append(str(shared_dir))

    def set_client(self, client_name: str) -> None:
        """この会議のクライアントに切り替える（タスクの行き先が変わる）。解決し直す。"""
        if client_name and client_name != self.cfg.client_name:
            self.cfg = replace(self.cfg, client_name=client_name)
            self._client = None

    def resolve_client(self) -> ClientRef:
        """クライアント参照を一度だけ解決して返す。"""
        if self._client is not None:
            return self._client
        try:
            module = importlib.import_module("task_hub_context")
            value = module.resolve_client(self.cfg.client_name)
        except (ImportError, AttributeError, OSError) as exc:
            raise TaskHubUnavailable("タスク管理 のクライアント解決モジュールを読み込めません") from exc
        if not value:
            raise TaskHubUnavailable(f"タスク管理 にクライアントがありません: {self.cfg.client_name}")
        try:
            self._client = ClientRef(
                client_id=str(value["client_id"]),
                client_name=str(value.get("client_name", self.cfg.client_name)),
                task_db_id=str(value["task_db_id"]),
                minutes_db_id=str(value["minutes_db_id"]),
            )
        except (KeyError, TypeError) as exc:
            raise TaskHubUnavailable("タスク管理 のクライアント情報が不完全です") from exc
        return self._client

    def register_task(self, todo: Item, brief_md: str, meeting_date: str, owner: str) -> dict:
        """TODO を Notion Tasks に登録して着手状態へ更新する。"""
        if not self.cfg.notion_tasks:
            return {"skipped": True}
        due_hint, deadline = self.due_hint_of(todo.due, meeting_date)
        task = {
            "summary": todo.text,
            "category": "other",
            "assignee": owner,
            "deadline": deadline,
            "due_hint": due_hint,
            "priority": "medium",
            "details": brief_md[:1800] + "（会議中に着手・realtime）",
            "target_article": None,
        }
        try:
            minutes = importlib.import_module("task_hub_minutes_lib")
            taskbrief = importlib.import_module("task_hub_taskbrief")
            registered = minutes.register_tasks(self.resolve_client().task_db_id, [task], meeting_date, None)
            value = registered[0]
            page_id = value["page_id"]
            taskbrief.set_task_status(page_id, "In Progress")
        except (ImportError, AttributeError, KeyError, IndexError, TypeError) as exc:
            raise TaskHubUnavailable("タスク管理 のタスク登録に失敗しました") from exc
        logger.info("タスク管理 task registered: %s", page_id)
        return {"page_id": page_id, "url": value.get("url", "")}

    def notify(self, text: str, category: str = "minutes") -> bool:
        """クライアントのチャット通知ワークフローへ投稿する。"""
        if not self.cfg.slack:
            return False
        try:
            response = httpx.post(
                self.cfg.chat_notify_url,
                json={"client_id": self.resolve_client().client_id, "text": text, "category": category},
                timeout=10,
            )
            if 200 <= response.status_code < 300:
                return True
            logger.warning("タスク管理 chat notification returned HTTP %s", response.status_code)
            return False
        except (httpx.HTTPError, TaskHubUnavailable, OSError) as exc:
            logger.warning("タスク管理 chat notification failed: %s", exc)
            return False

    @staticmethod
    def due_hint_of(due: str, meeting_date: str) -> tuple[str, str | None]:
        """曖昧な期日を タスク管理 が受ける決定論的な期限ヒントへ変換する。"""
        value = due.strip()
        if not value or value == "未定":
            return "none", None
        try:
            return "specific", date.fromisoformat(value).isoformat()
        except ValueError:
            pass
        try:
            month, day = (int(part) for part in value.split("/", 1))
            meeting = date.fromisoformat(meeting_date)
            deadline = date(meeting.year, month, day)
            if deadline < meeting:
                deadline = date(meeting.year + 1, month, day)
            return "specific", deadline.isoformat()
        except (ValueError, TypeError):
            pass
        hints = (("再来週", "two_weeks"), ("2 週間", "two_weeks"), ("今週", "this_week"), ("来週", "next_week"), ("今月", "this_month"), ("すぐ", "immediate"), ("至急", "immediate"))
        if value == "今日":
            return "specific", date.fromisoformat(meeting_date).isoformat()
        for keyword, hint in hints:
            if keyword in value:
                return hint, None
        return "none", None
