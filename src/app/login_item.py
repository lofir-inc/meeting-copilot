"""ログインしたときに常駐アプリを起こす（LaunchAgent の出し入れ）。

`open -a <アプリ>` を launchd に踏ませる。アプリの実行ファイルを直に起こすより、
  こちらのほうがマイク・画面収録の許可（TCC）がアプリに正しく結びつく。

入れっぱなしにしない: メニューから外せる。外したら plist を消す（無効化ではなく削除）。
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

LABEL = "com.meeting-copilot.menubar"


def plist_path(home: Path) -> Path:
    return Path(home) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def plist_bytes(app: Path) -> bytes:
    """launchd に渡す設定。常駐は 1 つだけ（KeepAlive は付けない）。"""
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": ["/usr/bin/open", "-a", str(app)],
        "RunAtLoad": True,
        "KeepAlive": False,          # 落ちたら落ちたまま。勝手に起こし直さない
    })


def is_enabled(home: Path) -> bool:
    return plist_path(home).is_file()


def enable(app: Path, home: Path, load=None) -> Path:
    """ログイン時に起こすようにする。既にあれば上書き（アプリの置き場所が変わることがある）。"""
    path = plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plist_bytes(Path(app)))
    (load or _load)(path)
    return path


def disable(home: Path, unload=None) -> bool:
    """やめる。無効化ではなくファイルごと消す（残骸を置かない）。"""
    path = plist_path(home)
    if not path.is_file():
        return False
    (unload or _unload)(path)
    path.unlink(missing_ok=True)
    return True


def _load(path: Path) -> None:                       # pragma: no cover - launchd を触る
    subprocess.run(["launchctl", "load", "-w", str(path)], capture_output=True)


def _unload(path: Path) -> None:                     # pragma: no cover - launchd を触る
    subprocess.run(["launchctl", "unload", "-w", str(path)], capture_output=True)
