"""会議の録音を、**聞き直せる小ささ**に畳む（マイク別＋統合した 1 本）。

2026-09-16 運用者 指示。実測: 118 分の会議で `recording_self.wav` と `recording_remote.wav` が
**各 649 MB**（合計 1.3 GB）。会議を重ねるとディスクを食う。文字起こしと議事録が済んだあとは、
**聞き直せれば十分**なので mp3 に畳む。

作るもの（`audio/` の下）:

| ファイル | 中身 | 使いどころ |
|---|---|---|
| `self.mp3` | 自分のマイクだけ | 自分の言い方を聞き直す |
| `remote.mp3` | 相手側だけ | 相手の発言を聞き直す・聞き取りにくい所を確かめる |
| `meeting.mp3` | 2 本を混ぜて音量をそろえた 1 本 | 通しで聞き返す・人に渡す |

元の wav は**消さない**（`keep_wav`）。作り直し（録音から全文を起こし直す）に要るため。
捨てるのは人が決める（`scripts/finish_meeting.py --drop-wav` でゴミ箱へ退避）。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
AUDIO_DIR = "audio"
SOURCES = {"self": "recording_self.wav", "remote": "recording_remote.wav"}
MERGED = "meeting.mp3"


@dataclass
class ArchiveConfig:
    """`settings.yaml` の `meeting.audio_archive`。"""

    enabled: bool = True
    bitrate: str = "64k"
    """声だけなので 64k で足りる（118 分で約 56 MB／wav の 1/11）。"""

    keep_wav: bool = True
    """元の wav を残すか。残さないと録音からの作り直しができない。"""

    normalize: bool = True
    """統合した 1 本の音量をそろえる（自分と相手で声の大きさが違うため）。"""

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ArchiveConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (values or {}).items() if key in known})


def compress(session_dir: Path, config: ArchiveConfig | None = None,
             *, runner=subprocess.run) -> dict:
    """録音を mp3 に畳む。既にあるものは作り直さない（何度呼んでもよい）。"""
    config = config or ArchiveConfig()
    session_dir = Path(session_dir)
    out_dir = session_dir / AUDIO_DIR
    made: dict[str, int] = {}
    sources: dict[str, Path] = {}
    for key, name in SOURCES.items():
        source = session_dir / name
        if not source.exists() or source.stat().st_size < 1024:
            continue
        sources[key] = source
        target = out_dir / f"{key}.mp3"
        if _fresh(target, source):
            made[key] = target.stat().st_size
            continue
        out_dir.mkdir(exist_ok=True)
        _run(runner, ["ffmpeg", "-nostdin", "-y", "-i", str(source), "-ac", "1",
                      "-b:a", config.bitrate, str(target)])
        made[key] = target.stat().st_size if target.exists() else 0
    if len(sources) == 2:
        merged = out_dir / MERGED
        newest = max(path.stat().st_mtime for path in sources.values())
        if not (merged.exists() and merged.stat().st_mtime >= newest):
            out_dir.mkdir(exist_ok=True)
            # 2 本を混ぜて 1 本に。normalize=0 で音が小さくなるのを防ぎ、loudnorm で声をそろえる
            chain = "amix=inputs=2:duration=longest:normalize=0"
            if config.normalize:
                chain += ",loudnorm=I=-16:TP=-1.5:LRA=11"
            _run(runner, ["ffmpeg", "-nostdin", "-y", "-i", str(sources["self"]), "-i", str(sources["remote"]),
                          "-filter_complex", chain, "-ac", "1", "-b:a", config.bitrate, str(merged)])
        if merged.exists():
            made["meeting"] = merged.stat().st_size
    return made


def wav_bytes(session_dir: Path) -> int:
    return sum((Path(session_dir) / name).stat().st_size
               for name in SOURCES.values() if (Path(session_dir) / name).exists())


def drop_wav(session_dir: Path, trash_dir: Path) -> list[Path]:
    """元の wav をゴミ箱へ退避する。mp3 が全部そろっているときだけ（無いのに捨てない）。"""
    session_dir, trash_dir = Path(session_dir), Path(trash_dir)
    present = [key for key in SOURCES if (session_dir / SOURCES[key]).exists()]
    if not present:
        return []
    missing = [key for key in present if not (session_dir / AUDIO_DIR / f"{key}.mp3").exists()]
    if missing:
        raise RuntimeError(f"mp3 がまだありません（{'・'.join(missing)}）。先に圧縮してください")
    target = trash_dir / f"{datetime.now(JST):%Y-%m-%d}_{session_dir.name}-wav"
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for key in present:
        source = session_dir / SOURCES[key]
        shutil.move(str(source), str(target / source.name))
        moved.append(target / source.name)
    (target / "WHY.md").write_text(
        f"# なぜここにあるか\n\n{session_dir.name} の録音（wav）。mp3（{AUDIO_DIR}/）に畳んだあと、"
        "容量のため退避した。録音から全文を作り直すにはこの wav が要る（mp3 からは起こし直さない）。\n",
        encoding="utf-8")
    logger.info("録音を退避しました: %s", target)
    return moved


def _fresh(target: Path, source: Path) -> bool:
    return target.exists() and target.stat().st_mtime >= source.stat().st_mtime and target.stat().st_size > 0


def _run(runner, argv: list[str]) -> None:
    result = runner(argv, capture_output=True, text=True)
    if getattr(result, "returncode", 0) != 0:
        raise RuntimeError(f"ffmpeg に失敗しました: {(result.stderr or '')[-200:]}")
