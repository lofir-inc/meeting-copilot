#!/usr/bin/env python3
"""MacWhisper が同じ会議を録った文字起こしを、評価用の「正解」JSONL として書き出す。

MacWhisper は Zoom 会議を自動で録音し、会議後に large-v3 で文字起こしして話者名まで付けている。
本システムの録音と同じ音を別経路で録っているので、**時刻を合わせれば比較の基準に使える**
（2026-09-11 の実会議で、相手側は 6 地点すべてでずれ 949.429 秒・相関 0.999 と一定だった）。

    python scripts/macwhisper_reference.py \\
        --recording workspace/sessions/2026-09-11-2/recording_remote.wav \\
        --out workspace/eval/2026-09-11-gen/reference.jsonl

時刻合わせは、こちらの録音から数か所を切り出して MacWhisper の録音の中を探す（相互相関）。
出力の start/end は **こちらの録音の先頭を 0 秒とする秒数**。MacWhisper 側の時刻ではない。
MacWhisper 起動中は DB が書き込み中なので、コピーしてから読む。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from statistics import median

import numpy as np

DB_DIR = Path.home() / "Library/Application Support/MacWhisper/Database"
SR = 16000


def _read_wav_16k_mono(path: Path) -> np.ndarray:
    """どの形式でも 16kHz モノラルの float 配列にして返す（ffmpeg 経由）。"""
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), str(converted)],
            check=True,
        )
        with wave.open(str(converted), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def find_offset(ours: np.ndarray, reference: np.ndarray, probes: int = 6, window_sec: float = 15.0) -> tuple[float, float]:
    """こちらの録音が、基準の録音の何秒目から始まるかを返す（オフセット秒, 相関の中央値）。"""
    results: list[tuple[float, float]] = []
    for index in range(probes):
        start = int(len(ours) * (index + 1) / (probes + 1))
        segment = ours[start : start + int(window_sec * SR)]
        if len(segment) < SR or float(np.sqrt(np.mean(segment**2))) < 1e-4:
            continue
        size = 1 << (len(reference) + len(segment) - 1).bit_length()
        correlation = np.fft.irfft(np.fft.rfft(reference, size) * np.conj(np.fft.rfft(segment, size)), size)
        correlation = correlation[: len(reference) - len(segment) + 1]
        # 移動エネルギーは累積和で出す（np.convolve は 1 時間の録音では終わらない）
        cumulative = np.concatenate(([0.0], np.cumsum(reference.astype(np.float64) ** 2)))
        window_energy = cumulative[len(segment) :] - cumulative[: -len(segment)]
        energy = np.sqrt(np.maximum(window_energy, 0.0)) * np.sqrt(np.sum(segment**2))
        # 基準側の**無音**はゼロ割れで相関が跳ね上がる（2026-09-14 に踏んだ: 相関 1907 という
        #   ありえない値が出て、無音の場所を「いちばん一致する」と選んでいた）。候補から外す。
        quiet = window_energy <= max(float(np.median(window_energy)) * 0.05, 1e-12)
        normalized = correlation / (energy + 1e-9)
        normalized[quiet] = 0.0
        best = int(np.argmax(normalized))
        results.append(((best - start) / SR, float(normalized[best])))
    if not results:
        raise SystemExit("時刻を合わせられませんでした（こちらの録音に音が入っていません）")

    # 地点ごとの答えが割れることがある（片方の録音が途中で切れている・別の会議を掴んでいる）。
    #   「いちばん多くの地点が同意した値」を採り、同意が足りなければ**書き出さずに止める**
    #   （ずれた基準で採点すると、数字だけがもっともらしく嘘になる）。
    offsets = sorted(offset for offset, _ in results)
    cluster: list[float] = []
    for candidate in offsets:
        agree = [offset for offset in offsets if abs(offset - candidate) <= 0.5]
        if len(agree) > len(cluster):
            cluster = agree
    agreed = len(cluster)
    scores = [score for offset, score in results if abs(offset - median(cluster)) <= 0.5]
    print(f"  時刻合わせ: {len(results)} 地点中 {agreed} 地点が一致"
          f"（相関 {median(scores):.3f}）", file=sys.stderr)
    if agreed < max(2, len(results) // 2):
        raise SystemExit(
            f"時刻を合わせられませんでした（{len(results)} 地点中 {agreed} 地点しか一致しません）。\n"
            "ずれた基準で採点すると数字が嘘になるので、書き出さずに止めます。\n"
            "  --kind mic-audio を試すか、対応する会議が合っているか確かめてください。"
        )
    return median(cluster), median(scores)


def _copy_db() -> Path:
    """書き込み中の DB をコピーして読み取り用のパスを返す。"""
    temp_dir = Path(tempfile.mkdtemp(prefix="macwhisper-db-"))
    for path in DB_DIR.glob("main.sqlite*"):
        shutil.copy2(path, temp_dir / path.name)
    copied = temp_dir / "main.sqlite"
    if not copied.exists():
        raise SystemExit(f"MacWhisper の DB が見つかりません: {DB_DIR}")
    return copied


def latest_session(db: sqlite3.Connection, on_date: str | None) -> tuple[bytes, str, str]:
    """（指定日の）いちばん新しい会議セッションの id・日時・録音 id を返す。"""
    where = "where r.date >= ? and r.date < date(?, '+1 day')" if on_date else ""
    args = (on_date, on_date) if on_date else ()
    row = db.execute(
        f"""
        select s.id, r.date, r.id from session s
        join recordedmeeting r on r.id = s.recordedMeetingID
        {where}
        order by r.date desc limit 1
        """,
        args,
    ).fetchone()
    if row is None:
        raise SystemExit("条件に合う MacWhisper の会議が見つかりません（日付は UTC で入っている）")
    return row[0], row[1], row[2]


def media_for(meeting_id: bytes, kind: str) -> Path:
    """録音 id と種別（app-audio / mic-audio）から音声ファイルを探す。"""
    prefix = meeting_id.hex().upper()
    prefix = f"{prefix[:8]}-{prefix[8:12]}-{prefix[12:16]}-{prefix[16:20]}-{prefix[20:]}"
    matches = sorted((DB_DIR / "ExternalMedia").glob(f"{prefix}_{kind}_*"))
    if not matches:
        raise SystemExit(f"{kind} の音声が見つかりません: {prefix}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", type=Path, required=True, help="こちらの録音（recording_remote.wav など）")
    parser.add_argument("--out", type=Path, required=True, help="書き出す正解 JSONL")
    parser.add_argument("--date", help="会議の日付（UTC・YYYY-MM-DD）。省略すると最新の会議")
    parser.add_argument("--kind", choices=["app-audio", "mic-audio"], default="app-audio", help="時刻合わせに使う MacWhisper 側の音声")
    args = parser.parse_args()

    db_path = _copy_db()
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    session_id, date, meeting_id = latest_session(db, args.date)
    print(f"MacWhisper の会議: {date} (UTC)")

    print("  時刻を合わせています…")
    offset, score = find_offset(_read_wav_16k_mono(args.recording), _read_wav_16k_mono(media_for(meeting_id, args.kind)))
    print(f"  こちらの録音は MacWhisper の {offset:.3f} 秒目から（相関 {score:.3f}）")

    rows = db.execute(
        """
        select t.start, t.end, t.text, sp.name from transcriptline t
        left join speaker sp on sp.id = t.speakerID
        where t.sessionId = ? order by t.start
        """,
        (session_id,),
    ).fetchall()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for start_ms, end_ms, text, speaker in rows:
            start = start_ms / 1000 - offset
            end = end_ms / 1000 - offset
            if end < 0 or not (text or "").strip():
                continue
            handle.write(json.dumps({"start_time": round(start, 3), "end_time": round(end, 3),
                                     "speaker": speaker or "不明", "text": text.strip()}, ensure_ascii=False) + "\n")
            written += 1
    print(f"  {written} 行を書き出しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
