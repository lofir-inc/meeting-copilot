#!/usr/bin/env python3
"""「会議アシスタント」を開く — 過去の会議の振り返り・仕上げの続き・辞書・声の台帳・会議を始める。

    python scripts/library_ui.py                                   # ブラウザで開く
    python scripts/library_ui.py --session 2026-09-14_1359         # その会議の候補を開く

会議が終わって候補があれば、自動でこの画面が開く（`meeting.open_library_after`）。
会議中はダッシュボードの「会議アシスタントを開く」から同じ画面が開く（こちらは起動しなくてよい）。
しばらく触らなければ自分で閉じる（`--idle-min`）。開きっぱなしのポートを残さない。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from src.app.user_path import ensure_user_path  # noqa: E402
from src.ui.library import Library, build_library_router, task_hub_pusher  # noqa: E402


def load_settings(path: Path | None) -> dict:
    for candidate in ([path] if path else []) + [REPO / "config" / "settings.yaml",
                                                  REPO / "config" / "settings.example.yaml"]:
        if candidate and candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return {}


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) == 0


def main() -> int:
    ensure_user_path()  # launchd から起こされる。仕上げ・議事録が ffmpeg / claude を探せるように
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session")
    parser.add_argument("--tab", default="meetings",
                        choices=["meetings", "finish", "candidates", "dictionary", "voices"])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--idle-min", type=float, default=60.0)
    args = parser.parse_args()

    host = "127.0.0.1"
    query = urlencode({key: value for key, value in (("tab", args.tab), ("session", args.session)) if value})
    url = f"http://{host}:{args.port}/library?{query}"
    if port_in_use(host, args.port):
        # もう開いている（前の会議のあとに開いたまま等）。二つ目は立てず、そちらを開く
        if not args.no_browser:
            webbrowser.open(url)
        print(f"既に開いています: {url}")
        return 0

    settings = lambda: load_settings(args.config)  # noqa: E731 — 画面を開くたびに読み直す（設定を直しても再起動不要）
    library = Library(REPO, settings, push_to_task_hub=task_hub_pusher(REPO, settings))
    app = FastAPI()
    app.include_router(build_library_router(library))
    last_seen = [time.monotonic()]

    @app.middleware("http")
    async def touch(request: Request, call_next):
        last_seen[0] = time.monotonic()
        return await call_next(request)

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/library")

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=args.port, log_level="warning"))

    def close_when_idle() -> None:
        while not server.should_exit:
            time.sleep(15)
            if time.monotonic() - last_seen[0] > args.idle_min * 60:
                print(f"{args.idle_min:.0f} 分触られなかったので閉じます")
                server.should_exit = True

    threading.Thread(target=close_when_idle, daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"会議アシスタント: {url}（{args.idle_min:.0f} 分触らなければ閉じます）")
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
