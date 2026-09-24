"""その会社の会議に**実際に出てくる人**を集める（議事録の誤変換候補の根拠にする）。

2026-09-18 に足した。ミナトの議事録に「社内（社長・北村氏）で検討」と書かれたが、
ミナトに北村という人は居なかった。調べると**文字起こしの段階で既に「北村さん」**で、
実際は「自分さん」（こちらの担当者）の聞き違いだった。

議事録は悪くない。`prompts/minutes.md` は「人名は全文の表記のまま書く・推測しない」と
決めてあり、そのとおり忠実に写しただけ。**勝手に直すほうが危ない**ので、この方針は変えない。

∴ 直すのではなく、**気づかせる**。その会社の会議に出てくる人の一覧を議事録の道具に渡し、
一覧に無い人名は「誤変換の候補」に挙げてもらう。人が見て決める。

**一覧に無い＝誤り、ではない。**初めて話に出た取引先の人は必ず一覧に無い。
だから「誤り」ではなく「候補」。そして拾った名前は次から一覧に入る（下の出どころが育つ）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HONORIFICS = ("さん", "氏", "様", "社長", "専務", "部長", "課長", "係長", "主任", "先生", "君")
"""名前の後ろに付く語。これが付いた語を「人名らしい」と見なす。"""

MENTION = re.compile(
    r"([一-龥ぁ-んァ-ヶーA-Za-z][^\s、。，．「」『』（）()\[\]・:：]{0,7}?)(" + "|".join(HONORIFICS) + r")(?![一-龥])"
)

NOT_NAMES = {
    "皆", "みな", "みんな", "お客", "おきゃく", "担当", "御社", "貴社", "弊社", "先方", "相手",
    "お子", "奥", "旦那", "うちの", "そちら", "こちら", "どちら", "何", "誰", "他", "ほか",
    "以上", "各", "関係", "参加", "出席", "責任", "代表",
}
"""人名ではないのに honorific が付く語。入れておかないと「お客さん」が人名になる。"""

TAIL = re.compile(r"[一-龥ァ-ヶーA-Za-z]+$")
"""名前の本体。前に付いた助詞や別の語を落とす。"""

MIN_LEN, MAX_NAMES = 2, 60


def mentions(text: str) -> set[str]:
    """本文から「人名らしい語」を拾う。根拠に使うだけで、置き換えには使わない。"""
    found = set()
    for stem, _honorific in MENTION.findall(text or ""):
        # 助詞を巻き込むので、末尾の漢字・カタカナ・英字のかたまりだけを採る
        #   （「件は西川さん」→「西川」。これをしないと「件は西川」が人名になる）
        tail = TAIL.search(stem)
        if tail is None:
            continue
        name = tail.group(0)
        # 「者」で終わるのは役割（担当者・責任者・関係者）。人名ではない
        if len(name) >= MIN_LEN and name not in NOT_NAMES and not name.endswith("者"):
            found.add(name)
    return found


def _client_of(session_dir: Path) -> str:
    try:
        return str(json.loads((session_dir / "client.json").read_text(encoding="utf-8")).get("name", ""))
    except (OSError, ValueError):
        return ""


def _speakers(session_dir: Path) -> set[str]:
    """その会議で**実際に話した**人の名前。「不明話者」は人名ではない。

    `speakers.json` は見ない（2026-09-18 に実データで判明）。声の登録は会社をまたいで
    引き継がれるので、**その会議に居なかった人まで入る**（さくら歯科の名簿に 参加者A が出た）。
    文字起こしに実際に出てきた話者なら、その会議で話した人だと言い切れる。
    """
    found = set()
    for name in ("transcripts_final.jsonl", "transcripts.jsonl"):
        path = session_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for token in re.findall(r'"speaker"\s*:\s*"([^"]+)"', text):
            if token and "不明話者" not in token:
                found.add(token)
        break                                   # 作り直した全文があればそちらだけ見る
    return found


def collect(sessions_dir: Path, client_name: str, *, extra: list[str] | None = None,
            limit: int = MAX_NAMES) -> list[str]:
    """その会社の会議に出てくる人を集める。

    出どころは 3 つ:

    1. 過去の会議で**名前を付けた話者**（`speakers.json` / `renames.jsonl`）
    2. 過去の議事録で**話に上がった人**（`minutes.md` の「〜さん」「〜氏」）
       ここが「所属者に限らない」を満たす（取引先・紹介者・同業者が入る）
    3. 呼び出し側が渡すもの（People Master のその会社の人など）

    会社が違う会議は見ない（`client.json` で絞る）。
    """
    # 会社名そのものが話者ラベルに入っていることがある（People Master の実データ）。人名ではない
    names: set[str] = {str(one).strip() for one in (extra or [])
                       if str(one).strip() and str(one).strip() != str(client_name).strip()}
    sessions_dir = Path(sessions_dir)
    if sessions_dir.is_dir() and client_name:
        for session in sessions_dir.iterdir():
            if not session.is_dir() or _client_of(session) != client_name:
                continue
            names |= _speakers(session)
            minutes = session / "minutes.md"
            if minutes.is_file():
                try:
                    names |= mentions(minutes.read_text(encoding="utf-8"))
                except OSError:
                    continue
    return sorted(names)[:limit]


def roster_block(names: list[str]) -> str:
    """議事録の道具へ渡す一文。空なら何も足さない（無い前提で書かせない）。"""
    if not names:
        return ""
    return (
        "\n\n## この会社の会議に出てくる人（誤変換の候補を見つけるため）\n\n"
        + "、".join(names)
        + "\n\n- この一覧は**過去の会議で名前が確かめられた人**です（所属者に限りません）。\n"
        "- 全文にこの一覧に**無い人名**が出てきたら、「誤変換の候補」に挙げてください"
        "（`reason` に「この会社の会議に出てきたことがない名前」と書く）。\n"
        "- **一覧に無い＝誤り、ではありません。**初めて話に出た人は必ず一覧にありません。\n"
        "  議事録の本文は**全文の表記のまま**書いてください。直すのは人です。\n"
    )
