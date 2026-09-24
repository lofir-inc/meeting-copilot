#!/usr/bin/env python3
"""メニューバーの常駐を始める（`会議アシスタント.app` の中身が呼ぶ）。

    python scripts/menubar_app.py

ふだんは直に叩かない。`scripts/アプリを作る.command` で作った .app から起動する。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.app import menubar  # noqa: E402
from src.app.user_path import ensure_user_path  # noqa: E402


def main() -> int:
    ensure_user_path()  # launchd の PATH は /usr/bin:/bin だけ。会議・画面はここから PATH を継ぐ
    return menubar.run(REPO, Path(sys.executable))


if __name__ == "__main__":
    raise SystemExit(main())
