#!/bin/bash
# ダブルクリックで会議モードを起動する（Finder / Dock から）。
#
# 入口はリポジトリ直下の `会議アシスタント.command` です。その画面の「会議を始める」ボタンが
#   これを開きます。急ぐときは、これを直接ダブルクリックしても構いません。
# このファイルは scripts/ に置いたまま使ってください（1 つ上の .venv を使います）。
#   デスクトップから開きたいときは、Finder で右クリック →「エイリアスを作成」。
set -u

REPO="${MEETING_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="$REPO/.venv/bin/python"

cd "$REPO" || { echo "リポジトリが見つかりません: $REPO"; read -r -p "Enter で閉じる"; exit 1; }

if [ ! -x "$VENV" ]; then
    echo "✗ Python の環境が見つかりません: $VENV"
    echo "  SETUP.md の «2. 取得と Python の環境» をもう一度実行してください。"
    read -r -p "Enter で閉じる"
    exit 1
fi

echo "会議モードを起動します（終わるときは画面の «会議を終了»）"
"$VENV" -m src.main --mode meeting_loopback
status=$?
echo
echo "終了しました（コード $status）。このウィンドウは閉じて構いません。"
read -r -p "Enter で閉じる"
