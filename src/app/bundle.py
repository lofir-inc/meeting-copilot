"""`会議アシスタント.app` を組み立てる。

なぜ手元で作るか（2026-09-18）: 署名していない .app を配布物に入れると、
  ダウンロードした人の Mac で Gatekeeper に隔離されて開けない。
  **手元で作った .app には隔離の印が付かない**ので、そのまま使える。

中身は Python スクリプト 1 本。作った時点の Python とリポジトリの場所を焼き込むので、
  使う人が動かしても本体を指し続ける。

**LaunchServices から起こされたプロセスのまま常駐してはいけない**（2026-09-18 実測）。
  Finder や `open -a` から起こしたプロセスがそのまま常駐すると、
  **メニューバーに状態項目が置かれない**（位置が x=0 のまま・`isVisible` は True を返し、
  プロセスも動き続けるので気づきにくい）。同じコードでも、ターミナルから直に起こすと置かれる。
  ∴ ここでは**切り離した子**として起こして、自分はすぐ終わる。

シェルを挟んで `exec` で別の python へ移らない。移すと、そのプロセスの「自分は誰か」が
  Python.framework の Python.app になる。

`LSUIElement` を true にして Dock に出さない（メニューバーだけに居る）。
"""

from __future__ import annotations

import plistlib
import stat
from pathlib import Path

APP_NAME = "会議アシスタント"
BUNDLE_ID = "com.meeting-copilot.menubar"
EXECUTABLE = "run"                                   # 中の実行ファイルは ASCII 名にする


def info_plist(name: str = APP_NAME, bundle_id: str = BUNDLE_ID) -> bytes:
    return plistlib.dumps({
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": bundle_id,
        "CFBundleExecutable": EXECUTABLE,
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSUIElement": True,                         # Dock に出さない
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
    })


def launcher(python: Path, repo: Path) -> str:
    """アプリの実行ファイル。切り離した子として常駐を起こし、自分はすぐ終わる（理由は上）。"""
    return (
        f"#!{python}\n"
        '"""会議アシスタント（メニューバー常駐）。scripts/アプリを作る.command が書き出す。\n'
        "\n"
        "この中身は書き換えないこと（作り直せば同じものが出る）。\n"
        "常駐そのものは scripts/menubar_app.py にある。ここは起こす役だけ。\n"
        '"""\n'
        "import subprocess\n"
        "\n"
        f"PYTHON = {str(python)!r}\n"
        f"REPO = {str(repo)!r}\n"
        "\n"
        "# 切り離して起こす。ここで常駐に入ると、メニューバーに状態項目が置かれない\n"
        "subprocess.Popen([PYTHON, REPO + \"/scripts/menubar_app.py\"], start_new_session=True)\n"
    )


def build(python: Path, repo: Path, into: Path, name: str = APP_NAME) -> Path:
    """`<into>/<name>.app` を作る。既にあれば作り直す。"""
    app = Path(into) / f"{name}.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "Info.plist").write_bytes(info_plist(name))
    binary = macos / EXECUTABLE
    binary.write_text(launcher(Path(python), Path(repo)), encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return app
