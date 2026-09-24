#!/usr/bin/env python3
"""過去の会議を、外（Gemini）の文字起こしで作り直す。元のセッションは触らない。

    ./scripts/rebuild_external.py

セッションを画面で選び、**別名の作業用セッション**を作って（録音はシンボリックリンク）
`finalize_meeting.py --external-stt` を回す。外へ出すかは途中で聞かれる。

なぜ別名にするか: `finalize_meeting.py` は `transcripts_final.jsonl` を上書きする。
元のセッションで回すと、**比べる相手（手元で作った版）が消える**。録音は数百 MB あるので
コピーせずリンクで済ませる。

引数を取らないのは、長い 1 行を端末で貼ると折り返しでコピーが壊れるため
（2026-09-13 に 3 度踏んだ）。選ぶのは画面で。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SESSIONS = REPO / "workspace" / "sessions"
LINKED = ("recording_remote.wav", "recording_self.wav")
COPIED = ("speakers.json", "corrections.jsonl")


def minutes(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate() / 60.0
    except Exception:
        return 0.0


def choose(candidates: list[Path]) -> Path | None:
    print("\n  作り直せる会議:")
    for index, session in enumerate(candidates, 1):
        length = minutes(session / "recording_remote.wav")
        print(f"    {index}. {session.name}（{length:.0f} 分）")
    print("    0. やめる")
    answer = input("\n  番号: ").strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(candidates):
        return None
    return candidates[int(answer) - 1]


def prepare(source: Path) -> Path:
    """録音をリンクした作業用セッションを作る（元は触らない）。"""
    work = SESSIONS / f"{source.name}-gemini"
    work.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        origin = source / name
        link = work / name
        if link.exists() or link.is_symlink():
            link.unlink()
        if origin.exists():
            link.symlink_to(origin.resolve())
    for name in COPIED:
        origin = source / name
        if origin.exists() and not (work / name).exists():
            shutil.copy2(origin, work / name)
    return work


def main() -> int:
    candidates = sorted(path for path in SESSIONS.iterdir()
                        if path.is_dir() and (path / "recording_remote.wav").exists())
    if not candidates:
        print("録音のあるセッションがありません")
        return 1
    if not sys.stdin.isatty():
        print("✗ 端末から実行してください（画面が無いと承認を聞けないので、送らずに終わります）")
        return 1

    source = choose(candidates)
    if source is None:
        print("  やめました")
        return 0

    work = prepare(source)
    print(f"\n  作業用セッション: {work}")
    print(f"  元の {source.name} は触りません（録音はリンク）")
    print("\n── 作り直しを回します（外へ出すかは途中で聞かれます）──")
    return subprocess.run([sys.executable, str(REPO / "scripts" / "finalize_meeting.py"),
                           str(work), "--external-stt"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
