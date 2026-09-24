"""`ffmpeg` があるかを、**使う前に**確かめる。

なぜ要るか（2026-09-20・まっさらな Mac を想定した見直しで発見）: `ffmpeg` が入っていないと、
会議のあとの作り直しが**生のスタックトレース**で落ちていた。

    FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'

しかもそのあと「議事録の材料がありません」と連鎖するので、**本当の原因が最後まで出てこない**。
SETUP には `brew install ffmpeg` と書いてあるが、書いてあることと、抜けたときに分かることは別。

会議**中**は ffmpeg を使わない（録音は wav で書く）。∴ 起動時は止めずに**警告だけ**にし、
  実際に使う手前で止める。会議の最中に落とさない、が優先。

PATH に Homebrew が無いときは足す（2026-09-24・本番で発生）: 画面（library_ui）やメニューバーは
  launchd 経由で起動され、PATH が `/usr/bin:/bin:/usr/sbin:/sbin` しか無い。`/opt/homebrew/bin/ffmpeg`
  が入っていても「見つからない」と判定され、「仕上げる」を押しても mp3 に畳まれなかった。
  ∴ 見つからなければ Homebrew の置き場所を探し、あれば**この処理の PATH の先頭に足す**。
  仕上げは最初に require() を通るので、そのあとの `["ffmpeg", …]` 呼び出し（子プロセス含む）にも効く。
"""

from __future__ import annotations

import os
import shutil

MISSING = (
    "ffmpeg が見つかりません。会議のあとの作り直しと録音の圧縮に必要です。\n"
    "  入れ方: brew install ffmpeg\n"
    "  （Homebrew が無ければ https://brew.sh のとおりに入れてください）"
)


# launchd 経由だと PATH に入っていない、Homebrew の置き場所（Apple シリコン／Intel）
SEARCH_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


def _add_homebrew_to_path() -> bool:
    for folder in SEARCH_DIRS:
        if os.access(os.path.join(folder, "ffmpeg"), os.X_OK):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


def available() -> bool:
    if shutil.which("ffmpeg") is not None:
        return True
    return _add_homebrew_to_path() and shutil.which("ffmpeg") is not None


def require(needed_for: str) -> None:
    """無ければ、分かる形で止める。トレースバックではなく、入れ方を出す。"""
    if available():
        return
    raise SystemExit(f"✗ {needed_for}には ffmpeg が要ります。\n\n{MISSING}")


def warn_if_missing(say=print) -> bool:
    """無ければ警告して False。会議そのものは回るので、止めない。"""
    if available():
        return True
    say(f"\n  ⚠ {MISSING}\n"
        "    会議は普通に回りますが、**終わったあとの作り直し・議事録・mp3 ができません**。")
    return False
