#!/usr/bin/env python3
"""外部文字起こしの経路を、合成音声のセッションで通しで見る。

    ./scripts/selftest_external.py

やること: 合成音声のセッションを作り直し → `finalize_meeting.py --external-stt` を回す。
途中で「外へ出してよいですか?」と聞かれる。端末から実行すること（画面が無いと聞けない）。

なぜ専用の入口を作るか: 引数つきの長い 1 行は端末で折り返し、コピーすると途中で
改行が混ざって別コマンドになる（2026-09-13 に 2 度踏んだ。`--external-stt` が
`command not found` になり、フラグ無しで走っていた）。貼る文字列を短く保つ。

見るのはこの 5 つ。精度ではなく**経路**:
  1. 会議ごとの確認が画面に出るか（出ないまま送っていたら重大な欠陥）
  2. `n` なら手元の Whisper に落ちて議事録ができるか
  3. `y` なら `external_sends.jsonl` に送った記録が残るか
  4. 話者の割り当てが手元で行われているか
  5. `(相づち)` の印が付き、辞書が効くか
     クラウド → Claude は直り、**クラウドファンディングは直らない**（守り札）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SESSION = "selftest-external"


def main() -> int:
    session = REPO / "workspace" / "sessions" / SESSION
    if not (session / "recording_remote.wav").exists():
        print("── 合成音声のセッションを作ります ──")
        made = subprocess.run([sys.executable, str(REPO / "scripts" / "make_test_session.py"),
                               "--name", SESSION])
        if made.returncode != 0:
            return made.returncode

    if not sys.stdin.isatty():
        print("✗ 端末から実行してください（画面が無いと承認を聞けないので、送らずに終わります）")
        return 1

    print("\n── 作り直しを回します（外へ出すかは途中で聞かれます）──")
    return subprocess.run([sys.executable, str(REPO / "scripts" / "finalize_meeting.py"),
                           str(session), "--external-stt"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
