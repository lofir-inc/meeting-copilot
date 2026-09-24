"""外で起こした文字起こしの区間に、手元の声紋で話者を当てる。

音声は 1 バイトも外へ出さない — 読むのは手元の録音だけ。外に任せるのは文字だけで、
「誰が言ったか」は手元に残す（運用者 方針 2026-09-13）。

2026-09-11 の実会議 50 分で測った結果（正解は MacWhisper）:

| 組み合わせ | 話者の一致 |
|---|---|
| Gemini の文字起こし ＋ Gemini の話者分け | 76% |
| ローカル Whisper ＋ resemblyzer（従来の本番） | 93% |
| **Gemini の文字起こし ＋ ここ** | **97%** |

正解の区切りで照合した上限も 97% なので、この組み合わせは既に上限に届いている。
Gemini は話者分けが弱いだけで、**区間の切り方は悪くない**。

`pad` が要る理由: 外の刻みは語単位まで細かく、565 区間のうち 311 件が `min_pool_sec`（3 秒）
未満だった＝声で判定する土俵に上がらない。前後を 0.4 秒伸ばすと全件が判定できるようになる。
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from src.audio.enrolled_diarizer import EnrolledDiarizer

logger = logging.getLogger(__name__)

DEFAULT_PAD_SEC = 0.4


def label_segments(
    rows: list[dict],
    audio: np.ndarray | None,
    sample_rate: int,
    diarizer: EnrolledDiarizer,
    *,
    pad_sec: float = DEFAULT_PAD_SEC,
    fallback: str = "不明話者1",
    slicer: Callable[[float, float], np.ndarray] | None = None,
    on_voice: Callable[[dict, np.ndarray], None] | None = None,
) -> dict[str, int]:
    """`rows` の `speaker` を、手元の声紋で判定し直す（その場で書き換える）。

    判定できなかった区間は**直前の話者を引き継ぐ**（無言で捨てない）。
    戻り値は話者名ごとの件数（`引き継ぎ` を含む）。

    `slicer` … 会議の時刻で音声を切り出す関数。会議中の 30 秒刻み（方式④）は**無音を落として
    繋いだ音声**を送るので、`audio` の先頭からの秒数では切り出せない（`src/stt/live_batch.py`
    の `AudioWindow.slice` を渡す）。省略すれば従来どおり `audio` を直線的に切る。
    `on_voice` … 行と、その行の声（embedding）を受け取る。会議中に画面で名前を直したとき、
    その声を覚えて過去の行へ反映するのに使う（`meeting_orchestrator._remember_voice`）。
    """
    counts: dict[str, int] = {}
    last = fallback
    total = len(audio) / sample_rate if (audio is not None and sample_rate) else 0.0
    for row in sorted(rows, key=lambda item: float(item["start_time"])):
        start = max(0.0, float(row["start_time"]) - pad_sec)
        end = float(row["end_time"]) + pad_sec
        if slicer is not None:
            chunk = slicer(start, end)
        else:
            chunk = audio[int(start * sample_rate):int(min(total, end) * sample_rate)]
        embedding = diarizer.embed(chunk, sample_rate) if on_voice is not None else None
        if embedding is not None:
            result = diarizer.identify_embedding(embedding, chunk.size / sample_rate, float(row["start_time"]))
        else:
            result = diarizer.identify(chunk, sample_rate, at=float(row["start_time"]))
        if not result.confident and result.similarity < 0:
            # 声では決められなかった区間。話の流れ（直前の話者）で埋める。
            name = last
            counts["引き継ぎ"] = counts.get("引き継ぎ", 0) + 1
        else:
            name = result.name
            last = name
        diarizer.report_text(name, len(str(row.get("text", ""))))
        row["speaker"] = name
        counts[name] = counts.get(name, 0) + 1
        if on_voice is not None and embedding is not None:
            on_voice(row, embedding)

    for old, new in diarizer.merge_unknowns():
        logger.info("不明話者クラスタを併合: %s → %s", old, new)
        for row in rows:
            if row["speaker"] == old:
                row["speaker"] = new
    return counts
