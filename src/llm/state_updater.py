"""Ollama を専用スレッドで呼ぶローリング状態更新器。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from src.llm.meeting_state import DELTA_SCHEMA, MeetingState, StateDelta, apply_delta
from src.llm.ollama_client import OllamaClient
from src.stt.whisper_client import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class StateUpdaterConfig:
    """ローリング状態更新の設定。"""

    interval_sec: float = 30.0
    window_max_sec: float = 120.0
    max_prompt_chars: int = 12000
    num_ctx: int = 8192
    num_predict: int = 700
    think: bool = False
    max_attempts: int = 3
    prior_tasks_chars: int = 2000
    """事前資料のうち、毎回の窓のプロンプトへ入れる先頭の文字数。

    全文は入らない。画面から資料を足したとき、どこまでが要約に渡っているかは
    ダッシュボードの「事前資料」に出す（黙って切られるのがいちばん困るため）。
    """
    max_carry_rounds: int = 3
    """反映できなかった発話を持ち越す回数の上限。超えたら捨てる（同じ窓で毎回 JSON が壊れると、後ろが全部詰まるため）。"""


@dataclass
class UpdateStats:
    """1 回の状態更新で得た計測値。"""

    prompt_chars: int = 0
    latency_sec: float = 0.0
    parse_ok: bool = True
    retried: bool = False
    changes: list[str] = field(default_factory=list)
    leftover: list[TranscriptSegment] = field(default_factory=list)
    """反映できずに残った発話。StateLoop が次の回へ持ち越す。"""


def _validate_delta(data: dict) -> StateDelta:
    """JSON Schema と同等の最小検証をして StateDelta に変換する。"""
    required = set(DELTA_SCHEMA["required"])
    if set(data) != required:
        raise ValueError("差分 JSON のキーが schema と一致しません")
    list_keys = ("new_decisions", "new_todos", "new_questions", "updates", "next_asks")
    if not isinstance(data["summary"], str) or not isinstance(data["current_topic"], str):
        raise ValueError("summary または current_topic が文字列ではありません")
    if not isinstance(data["topic_changed"], bool):
        raise ValueError("topic_changed が真偽値ではありません")
    if any(not isinstance(data[key], list) for key in list_keys) or len(data["next_asks"]) > 3:
        raise ValueError("差分 JSON の配列が不正です")
    shapes = {
        "new_decisions": ({"text", "by"}, None),
        "new_todos": ({"text", "owner", "due"}, None),
        "new_questions": ({"text", "asked_by"}, None),
        "updates": ({"id", "status", "note"}, {"done", "dropped", "answered", "superseded", "closed", "asked"}),
    }
    for key, (keys, statuses) in shapes.items():
        for value in data[key]:
            if not isinstance(value, dict) or set(value) != keys or not all(isinstance(entry, str) for entry in value.values()):
                raise ValueError(f"{key} の要素が schema と一致しません")
            if statuses is not None and value["status"] not in statuses:
                raise ValueError("updates.status が不正です")
    if not all(isinstance(value, str) for value in data["next_asks"]):
        raise ValueError("next_asks の要素が文字列ではありません")
    return StateDelta(**data)


class StateUpdater:
    """会議の状態を LLM の差分応答で更新する。"""

    def __init__(self, client: OllamaClient, system_prompt: str, prior_tasks: str, cfg: StateUpdaterConfig) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.prior_tasks = prior_tasks
        self.cfg = cfg

    def build_prompt(self, prev: MeetingState, window: list[TranscriptSegment]) -> str:
        """前回状態と時刻順の発話からユーザープロンプトを作る。"""
        transcript = "\n".join(
            f"[{int(segment.start_time // 60)}:{int(segment.start_time % 60):02d}] {segment.speaker}: {segment.text}"
            for segment in sorted(window, key=lambda value: value.start_time)
        )
        prefix = (
            f"## 前回タスク一覧\n{self.prior_tasks[:self.cfg.prior_tasks_chars] or '（なし）'}\n\n"
            f"## 現在の状態\n{prev.to_prompt_text()}\n\n## 直近の発話\n"
        )
        suffix = "\n\n## 出力の指示\n会話から得られた差分だけを、指定された JSON Schema に厳密に従って返してください。"
        if len(prefix) + len(suffix) > self.cfg.max_prompt_chars:
            prefix = prefix[:max(0, self.cfg.max_prompt_chars - len(suffix))]
        available = max(0, self.cfg.max_prompt_chars - len(prefix) - len(suffix))
        return prefix + (transcript or "（発話なし）")[:available] + suffix

    def _update_once(self, prev: MeetingState, window: list[TranscriptSegment]) -> tuple[MeetingState, UpdateStats]:
        """1 つのウィンドウを一度だけ更新する。"""
        prompt = self.build_prompt(prev, window)
        started = time.monotonic()
        data = self.client.chat_json(
            self.system_prompt,
            prompt,
            DELTA_SCHEMA,
            num_ctx=self.cfg.num_ctx,
            num_predict=self.cfg.num_predict,
            think=self.cfg.think,
        )
        delta = _validate_delta(data)
        at = max((segment.end_time for segment in window), default=prev.updated_at)
        state, changes = apply_delta(prev, delta, at)
        return state, UpdateStats(prompt_chars=len(prompt), latency_sec=time.monotonic() - started, changes=changes)

    def _windows(self, window: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
        """会議秒の幅が大きいウィンドウを指定幅で分割する。"""
        ordered = sorted(window, key=lambda value: value.start_time)
        if not ordered:
            return [ordered]
        groups: list[list[TranscriptSegment]] = [[]]
        start = ordered[0].start_time
        for segment in ordered:
            if groups[-1] and segment.start_time - start > self.cfg.window_max_sec:
                groups.append([])
                start = segment.start_time
            groups[-1].append(segment)
        return groups

    def update(self, prev: MeetingState, window: list[TranscriptSegment]) -> tuple[MeetingState, UpdateStats]:
        """遅延時の分割と設定回数の再試行を含めて状態を更新する。"""
        state = prev
        total = UpdateStats()
        parts = self._windows(window)
        for index, part in enumerate(parts):
            for attempt in range(self.cfg.max_attempts):
                if attempt >= 1:
                    total.retried = True
                try:
                    state, stats = self._update_once(state, part)
                    total.prompt_chars = max(total.prompt_chars, stats.prompt_chars)
                    total.latency_sec += stats.latency_sec
                    total.changes.extend(stats.changes)
                    break
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("状態更新に失敗しました (%d/%d): %s", attempt + 1, self.cfg.max_attempts, exc)
                    if attempt == self.cfg.max_attempts - 1:
                        # 捨てない（2026-09-11 本番: Ollama が 12 分壊れていた間の発話が要約から丸ごと消えた）。
                        #   反映済みの前半は残し、失敗した部分から後ろを呼び出し側へ返す
                        total.parse_ok = False
                        total.leftover = [segment for rest in parts[index:] for segment in rest]
                        return state, total
        return state, total


class StateLoop(threading.Thread):
    """オーケストレーターから受けた発話を一定間隔で状態更新する。"""

    def __init__(
        self,
        updater: StateUpdater,
        initial_state: MeetingState | None = None,
        on_state: Callable[[MeetingState, UpdateStats, list[str]], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.updater = updater
        self.on_state = on_state
        self._state = initial_state or MeetingState()
        self._buffer: list[TranscriptSegment] = []
        self._carried_rounds = 0
        self._lock = threading.Lock()
        self._processing = threading.Lock()
        """update は 1 本ずつ。run() が LLM 呼び出し中に flush() が走ると、同じ前回状態から
        2 本の更新が出て片方が失われる（lost update）。"""
        self._wake = threading.Event()
        self._stopping = threading.Event()

    def push(self, seg: TranscriptSegment) -> None:
        """文字起こし済み発話をスレッド安全に蓄積する。"""
        with self._lock:
            self._buffer.append(seg)

    def state(self) -> MeetingState:
        """現在状態の独立したスナップショットを返す。"""
        with self._lock:
            return MeetingState.from_json(self._state.to_json())

    def _take_buffer(self) -> list[TranscriptSegment]:
        """現在のバッファを取り出す。"""
        with self._lock:
            window, self._buffer = self._buffer, []
        return window

    def _process(self, window: list[TranscriptSegment]) -> MeetingState:
        """取り出した発話を更新し、フックを呼び出す。"""
        if not window:
            return self.state()
        with self._processing:
            previous = self.state()
            state, stats = self.updater.update(previous, window)
            with self._lock:
                self._state = state
                if not stats.leftover:
                    self._carried_rounds = 0
                elif self._carried_rounds < self.updater.cfg.max_carry_rounds:
                    # 次の回に、その間に届いた発話より前へ戻す（時刻順を保つ）
                    self._carried_rounds += 1
                    self._buffer = stats.leftover + self._buffer
                else:
                    logger.warning("状態更新に %d 回続けて失敗したので、%d 発話を要約に入れずに捨てます", self._carried_rounds + 1, len(stats.leftover))
                    self._carried_rounds = 0
            if self.on_state is not None:
                self.on_state(state, stats, stats.changes)
        return state

    def flush(self, timeout: float) -> MeetingState:
        """残りの発話を同期的に処理して状態を返す。"""
        del timeout
        state = self._process(self._take_buffer())
        with self._lock:
            # 会議終了の確定処理: 聞かれないまま残った問いを skipped にする（途中では open のまま）
            for change in self._state.finalize(self._state.updated_at):
                logger.info("finalize %s", change)
            return MeetingState.from_json(self._state.to_json())

    def stop(self) -> None:
        """ループを終了する。"""
        self._stopping.set()
        self._wake.set()

    def run(self) -> None:
        """interval ごとにバッファを取り出して更新する。"""
        while not self._stopping.is_set():
            self._wake.wait(self.updater.cfg.interval_sec)
            self._wake.clear()
            if not self._stopping.is_set():
                self._process(self._take_buffer())
