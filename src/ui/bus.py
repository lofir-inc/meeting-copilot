"""ローカル UI 向けのイベント配信を扱う。"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal

logger = logging.getLogger(__name__)

EventType = Literal["segment", "state", "level", "status", "rename", "relabel", "dissolve", "metrics", "participants", "dispatch", "factcheck", "preflight", "finish"]
_HISTORY_TYPES = {"segment", "state", "rename", "relabel", "dissolve", "dispatch", "factcheck", "preflight", "finish"}
"""あとから画面を開いた人にも見せる種別（ファクトチェックは会議の途中から見ても要る）。"""
_QUEUE_SIZE = 10_000


@dataclass
class Event:
    """UI に送る単一のイベントを表す。"""

    type: EventType
    seq: int
    at: float
    data: dict

    def to_json(self) -> str:
        """SSE の data 行へ載せる JSON 文字列を返す。"""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


class Subscription:
    """EventBus の一購読者を表す。"""

    def __init__(self, bus: "EventBus") -> None:
        self._bus = bus
        self._queue: queue.Queue[Event] = queue.Queue(maxsize=_QUEUE_SIZE)
        self._closed = False

    def get(self, timeout: float | None = None) -> Event | None:
        """指定時間まで次のイベントを待ち、なければ None を返す。"""
        if self._closed:
            return None
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """購読を解除する。"""
        with self._bus._lock:
            if self._closed:
                return
            self._closed = True
            self._bus._subscriptions.discard(self)

    def _put(self, event: Event) -> None:
        """イベントを追加し、満杯なら最古のイベントを捨てる。"""
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("UI event queue remained full while publishing event: %s", event.type)
        else:
            logger.warning("UI event queue was full; dropped its oldest event")


class EventBus:
    """履歴再生とライブ配信を両立するスレッド安全なイベントバス。"""

    def __init__(self, history: int = 5000) -> None:
        self._lock = threading.Lock()
        self._history: deque[Event] = deque(maxlen=history)
        self._latest: dict[str, Event] = {}
        self._subscriptions: set[Subscription] = set()
        self._seq = 0

    def publish(self, type: EventType, data: dict) -> Event:
        """イベントを採番して購読者へ配信する。"""
        with self._lock:
            self._seq += 1
            event = Event(type=type, seq=self._seq, at=time.time(), data=data)
            self._latest[type] = event
            if type in _HISTORY_TYPES:
                self._history.append(event)
            for subscription in tuple(self._subscriptions):
                subscription._put(event)
        return event

    def subscribe(self) -> Subscription:
        """履歴を先に渡した新しい購読を返す。"""
        with self._lock:
            subscription = Subscription(self)
            for event in self._history:
                subscription._put(event)
            self._subscriptions.add(subscription)
        return subscription

    def latest(self, type: str) -> Event | None:
        """指定種別の直近イベントを返す。"""
        with self._lock:
            return self._latest.get(type)
