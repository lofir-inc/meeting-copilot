#!/usr/bin/env python3
"""登録済みクライアントの一覧を Notion（Client Master）から取り込んで手元に控える。

    python scripts/sync_clients.py          # 一覧を取り込んで表示

会議の画面の「この会議の相手」はこの控えを出す。控えがあれば Notion に繋がらない日でも選べる。
会議の終わりにも自動で読み直す（`src/meeting_orchestrator.py`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import clients  # noqa: E402


def main() -> int:
    settings_path = REPO / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    shared = ((settings or {}).get("task_hub") or {}).get("shared_dir", "~/.claude/skills/_shared")
    try:
        people_db = str((settings.get("task_hub") or {}).get("people_master_db", ""))
        found = clients.fetch(shared, people_db=people_db)
    except Exception as error:  # noqa: BLE001
        print(f"✗ 一覧を読めませんでした: {error}")
        return 1
    clients.save_cache(REPO / clients.CACHE_FILE, found)
    for option in clients.options(found):
        mark = "（テスト）" if option["test"] else ""
        print(f"  {option['name']}（{option['client_id'] or '-'}）"
              f"{'' if option['has_minutes_db'] else ' 議事録DBなし'}{mark}")
    print(f"→ {REPO / clients.CACHE_FILE}（{len(found)} 件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
