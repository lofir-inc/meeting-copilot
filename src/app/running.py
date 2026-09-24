"""いま会議が動いているかを**見るだけ**で調べる。

なぜポートを見るか（2026-09-18）: 会議の経路には一切手を入れない、と決めたため。
  常駐アプリのために `meeting_orchestrator` に状態ファイルを書かせると、
  **会議中に落ちる経路が 1 つ増える**。見るだけなら、間違っても会議は止まらない。

ダッシュボードは 8765 が塞がっていると隣のポートへ逃げる（`src/ui/server.py` の
  `PORT_TRIES = 10`）。だから 8765〜8774 を順に見る。
会議以外のものが 8765 を使っていることがある（2026-09-17 に別プロジェクトの
  `python -m http.server` が居座った）。`session_name` が返るかどうかで見分ける。
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime

HOST = "127.0.0.1"
DASHBOARD_PORTS = range(8765, 8775)
"""会議のダッシュボードが使いうるポート（`src/ui/server.py` の PORT_TRIES と同じ幅）。"""

LIBRARY_PORT = 8766
"""会議アシスタント。会議中はダッシュボードがここまで使うことがある。"""


@dataclass
class Meeting:
    """動いている会議。"""

    port: int
    session_name: str
    started_at: datetime | None = None
    """録音の先頭。`/api/status` の `started_at`（時差つきの ISO）を読んだもの。

    2026-09-18: 最初 `elapsed_sec` を読むつもりで書いたが、`session_info()` が返すのは
    `started_at` だった。経過はこちらで引き算する。
    """

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"


def port_open(port: int, host: str = HOST, timeout: float = 0.15) -> bool:
    """そのポートで誰かが待ち受けているか。中身は見ない（速い）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def _ask(port: int, timeout: float = 0.6) -> dict | None:
    """`/api/status` に聞く。会議でなければ None（例外にしない）。"""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/status", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def find_meeting(ports=DASHBOARD_PORTS, is_open=port_open, ask=_ask) -> Meeting | None:
    """動いている会議を 1 つ探す。無ければ None。

    見つけた最初の 1 つで止める（同じ Mac で 2 つ会議を回すことは想定しない）。
    """
    for port in ports:
        if not is_open(port):
            continue
        payload = ask(port)
        if not payload:
            continue                      # 会議以外のものが使っている
        name = str(payload.get("session_name") or "").strip()
        if not name:
            continue
        return Meeting(port=port, session_name=name, started_at=_started(payload.get("started_at")))
    return None


def _started(told) -> datetime | None:
    """`started_at` を読む。読めなければ None（経過を出さないだけ）。"""
    try:
        return datetime.fromisoformat(str(told))
    except (TypeError, ValueError):
        return None


def title(meeting: Meeting | None) -> str:
    """メニューバーに出す字。記録中はひと目で分かるようにする。"""
    return "🔴" if meeting else "🎙"


def status_line(meeting: Meeting | None, now: datetime | None = None) -> str:
    """メニューの 1 行目。押せない見出しとして出す。"""
    if meeting is None:
        return "待機中"
    if meeting.started_at is None:
        return f"記録中: {meeting.session_name}"
    now = now or datetime.now(meeting.started_at.tzinfo)
    minutes = max(0, int((now - meeting.started_at).total_seconds() // 60))
    if minutes < 60:
        return f"記録中: {meeting.session_name}（{minutes} 分）"
    return f"記録中: {meeting.session_name}（{minutes // 60} 時間 {minutes % 60} 分）"
