#!/bin/bash
# ダブルクリックで「会議アシスタント」を開く。
#   過去の会議（振り返り・聞き直す・議事録）／会議のあとの仕上げ／辞書／声の台帳／会議を始める
# 会議を始めるときも、まずここを開いてよい（「会議を始める」ボタンがある）。60 分触らなければ閉じる（`--idle-min` の既定）。
set -u
REPO="${MEETING_REPO:-$(cd "$(dirname "$0")" && pwd)}"
PYTHON="$REPO/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
exec "$PYTHON" "$REPO/scripts/library_ui.py" "$@"
