"""相づちだけの行に印を付ける（落とさずに残す）。

なぜ落とさないか（運用者 決定 2026-09-13）: 相づちは「聞いていた」「同意した」の記録でもあり、
どこで相手が引っかかったかが読める。消してしまうと後から戻せない。**印を付けて残す**。

なぜ要るか: 外（Gemini）の文字起こしは相づちを 1 つも取りこぼさない代わりに、全部を 1 行として
出す。2026-09-11 の会議 50 分では 945 行のうち **「うん。」203 行・「はい。」77 行・
「うんうん。」30 行・「うんうんうん。」21 行・「ああ。」18 行** が並んだ。
ローカル Whisper との食い違い 41% 対 30% の差も、中身はほぼこれ（相づちを除くと 28.9% 対 26.5%）。
議事録を作る側がこの行を読み飛ばせるように、機械で分かる印を付ける。

判定は**保守的に**する — 行の全部が相づち語の繰り返しのときだけ。
「うん、確かに。」「はい。ぜひぜひ。」「そうです。」は中身がある側に残す。
"""

from __future__ import annotations

import re
import unicodedata

# 実データ（2026-09-11 の会議 50 分・8 文字以下の行 504 件）から拾った語。
# 長いものから剥がすので、並び順は意味を持つ（`_TOKENS` はソートして使う）。
_TOKENS = (
    "なるほど", "うんうん", "そうそう", "はいはい",
    "うーん", "ふーん", "えーと", "えっと", "うむ",
    "うん", "はい", "ああ", "あー", "ええ", "えー", "おお", "おー",
    "そう", "へえ", "へー", "はあ", "はー", "ほお", "ほー",
    "まあ", "まー", "あの", "ふん", "あ", "ん",
)

# 句読点・記号・空白は判定の前に落とす
_NOISE = re.compile(r"[\s、。，．,.!?！？…‥・「」『』（）()\-ー〜~]+")

_SORTED = tuple(sorted(_TOKENS, key=len, reverse=True))


def is_aizuchi(text: str) -> bool:
    """その行が相づち（とつなぎ言葉）だけでできているか。

    空文字は False（相づちではなく「何も無い」なので、別の扱いをする）。
    """
    stripped = _NOISE.sub("", unicodedata.normalize("NFKC", text))
    if not stripped:
        return False
    while stripped:
        for token in _SORTED:
            if stripped.startswith(token):
                stripped = stripped[len(token):]
                break
        else:
            return False
    return True


def mark_rows(rows: list[dict], *, enabled: bool = True) -> int:
    """各行に `aizuchi` の印を付け、付いた数を返す。

    印を付けるだけで、行は 1 つも消さない。
    """
    if not enabled:
        return 0
    marked = 0
    for row in rows:
        flag = is_aizuchi(str(row.get("text", "")))
        row["aizuchi"] = flag
        marked += int(flag)
    return marked
