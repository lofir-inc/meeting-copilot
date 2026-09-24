"""会議を**ターミナルを出さずに**起こす。

2026-09-18 運用者 指摘「ターミナルが立ち上がって、会議終了後にも Enter で閉じないと
  いけないのは良くない。ターミナルは一切見えないようになるといいね」。

端末が要らないことは確かめてある: 会議中に入力を待つ 3 か所（起動セルフチェックの確認・
  外へ出す確認・測定の開始）は、どれも**画面が無いときだけ**端末を使う。話者登録は
  `meeting.skip_enrollment` で飛ばしている。∴ 画面さえ立てば端末は一度も要らない。

ただし**黙って死なせない**。切り離すと画面にも端末にも何も出ないので、
  出力は必ずファイルに落とし、立ち上がらなかったら呼び手が中身を見せる。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

LOG_NAME = "last_meeting.log"
"""会議の起動ログ。毎回まっさらにする（前の回の失敗と混ざると読めない）。"""

WAIT_SEC = 90.0
"""画面が立つまで待つ上限。モデルの読み込みで 1 分近くかかることがある。"""


def log_path(repo: Path) -> Path:
    return Path(repo) / "workspace" / LOG_NAME


def start(repo: Path, python: Path, popen=None) -> subprocess.Popen:
    """会議を切り離して起こす。親（常駐アプリ）が終わっても会議は続く。

    `popen` は呼ぶときに決める（既定引数で束ねると、差し替えが効かない）。
    """
    popen = popen or subprocess.Popen
    repo, python = Path(repo), Path(python)
    path = log_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", buffering=1)
    return popen([str(python), "-m", "src.main", "--mode", "meeting_loopback"],
                 cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT,
                 stdin=subprocess.DEVNULL, start_new_session=True)


def tail(repo: Path, lines: int = 12) -> str:
    """起動ログの末尾。立ち上がらなかったときに、そのまま画面へ出す。"""
    try:
        text = log_path(repo).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "（起動ログがありません）"
    kept = [line for line in text.splitlines() if line.strip()][-lines:]
    return "\n".join(kept) or "（起動ログが空です）"


def why_not_started(process: subprocess.Popen | None, repo: Path) -> str:
    """立ち上がらなかった理由を、そのまま人に見せる文にする。"""
    code = process.poll() if process is not None else None
    head = ("会議が終了しています（終了コード %s）" % code) if code is not None \
        else "会議の画面が %.0f 秒たっても開きませんでした" % WAIT_SEC
    return f"{head}\n\n{tail(repo)}"
