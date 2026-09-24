"""窓の切れ目で割れた文を、読める形につなぎ直す。

2026-09-18 の実会議（Deepgram）で出た形:

    [自分]  自分のマイクです
    [自分]  。こんにちは、お世話になります。
    [長谷川] 先日はどうも。…お話しありましょう
    [長谷川] 。今日

30 秒の窓で文の途中が切れると、**前の文の句点が次の窓の先頭に落ちる**。
外のエンジンは窓ごとに独立して句読点を付けるので、こちらで戻すしかない。

戻すのは「行の先頭に立つ閉じ記号」だけ。本文の中身は触らない。
"""

from __future__ import annotations

LEADING = "。．、，？！?!…」』）)"
"""行の先頭に来たら前の行の末尾へ返す記号。開き括弧は含めない（そこで始まる文がある）。"""

CLOSERS = "。．？！?!"
"""これで終わっていれば、句点をさらに足さない。"""


def pull_back(text: str) -> tuple[str, str]:
    """行の先頭に立つ閉じ記号を切り出す。返り値は (前の行へ返すぶん, 残り)。

    >>> pull_back("。こんにちは")
    ('。', 'こんにちは')
    >>> pull_back("こんにちは")
    ('', 'こんにちは')
    """
    text = text or ""
    cut = 0
    while cut < len(text) and text[cut] in LEADING:
        cut += 1
    return text[:cut], text[cut:].lstrip()


def stitch(previous: str, current: str) -> tuple[str, str]:
    """1 組ぶんつなぎ直す。返り値は (直した前の行, 直したいまの行)。

    いまの行が記号だけだった場合、空文字を返す（呼び手が行ごと捨てられるように）。
    """
    moved, rest = pull_back(current)
    if not moved:
        return previous, current
    previous = (previous or "").rstrip()
    if previous and previous[-1] in CLOSERS:
        moved = moved.lstrip("。．、，")      # 句点が二重にならないようにする
    return previous + moved, rest
