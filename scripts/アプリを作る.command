#!/bin/bash
# ダブルクリックで「会議アシスタント.app」を作る（メニューバーに常駐する版）。
#
# なぜ手元で作るか: 署名していない .app を配った場合、ダウンロードした Mac では
#   Gatekeeper に隔離されて開けない。手元で作った .app には隔離の印が付かないので、
#   そのまま使える。
# 作り直しても設定は消えない（アプリは入れ物で、中身はこのリポジトリを指しているだけ）。
# 置き場所は /Applications。Finder の「アプリケーション」から Dock へドラッグできる。
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# リポジトリ直下に作った .venv を使う（SETUP.md 2 のとおり）。
VENV="$REPO/.venv/bin/python"

if [ ! -x "$VENV" ]; then
    echo "✗ Python の環境が見つかりません: $VENV"
    echo "  SETUP.md の «2. 取得と Python の環境» をもう一度実行してください。"
    read -r -p "Enter で閉じる"
    exit 1
fi

echo "=================================================="
echo " 会議アシスタント.app を作ります"
echo "=================================================="
echo " 置き場所 : /Applications"
echo " 本体     : $REPO"
echo

"$VENV" - "$REPO" "$VENV" <<'PY'
import sys
from pathlib import Path

repo, python = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from src.app import bundle

target = Path("/Applications")
if not target.is_dir():
    target = Path.home() / "Applications"
    target.mkdir(parents=True, exist_ok=True)
app = bundle.build(python, repo, target)
print(f"できました: {app}")
PY
status=$?

echo
if [ "$status" -eq 0 ]; then
    echo "メニューバーに 🎙 が出ます（会議を記録している間は 🔴）。"
    echo "できることは、会議を始める／会議の画面を開く／会議アシスタントを開く／ログイン時に起動する。"
    echo
    read -r -p "いま起動しますか? [y/N] " answer
    case "$answer" in
        [yY]*) open -a "/Applications/会議アシスタント.app" 2>/dev/null \
               || open -a "$HOME/Applications/会議アシスタント.app" ;;
    esac
else
    echo "⚠ 作れませんでした（終了コード $status）。上のログを確認してください。"
fi
read -r -p "Enter で閉じる"
