"""`.app` や launchd から起こされたとき、ターミナルと同じ置き場所のコマンドが見えるようにする。

なぜ要るか（2026-09-24・本番で発生）: `会議アシスタント.app`・メニューバー・画面（library_ui）は
  launchd 経由で起動され、PATH が `/usr/bin:/bin:/usr/sbin:/sbin` しか無い。
  ffmpeg が「見つからない」で仕上げが止まった（`src/audio/ffmpeg.py` で個別に直した）が、
  同じ理由で `claude`（最終パス）・`ollama`・`gcloud`（課金の確認）も見えなくなる。
  ∴ 入口（常駐・画面・会議本体）で**まとめて**足す。個別の探し直しは ffmpeg にだけ残す。

足すのは「実在するフォルダ」だけ。既にあるものは足さない（順番を変えない）。
後ろに足す。ターミナルから起こしたとき、その人の PATH の優先順を崩さない。
"""

from __future__ import annotations

import os
from pathlib import Path

CANDIDATES = (
    "/opt/homebrew/bin",      # Homebrew（Apple シリコン）
    "/usr/local/bin",         # Homebrew（Intel）・公式インストーラの多く
    "~/.local/bin",           # Claude CLI の既定の置き場所
)


def ensure_user_path(environ=None) -> list[str]:
    """足したフォルダを返す。`environ` は差し替え用（既定は os.environ）。"""
    environ = os.environ if environ is None else environ
    current = [p for p in environ.get("PATH", "").split(os.pathsep) if p]
    added = []
    for raw in CANDIDATES:
        folder = str(Path(raw).expanduser())
        if folder in current or not Path(folder).is_dir():
            continue
        current.append(folder)
        added.append(folder)
    if added:
        environ["PATH"] = os.pathsep.join(current)
    return added
