#!/usr/bin/env python3
"""声の台帳を見る・過去の会議から覚える・名前を直す・外す。すべて手元だけで動く（何も送らない）。

    python scripts/voice_library.py list
    python scripts/voice_library.py add workspace/sessions/<会議> [<会議> ...]   # 過去の会議から覚える
    python scripts/voice_library.py rename "Headset Shuting Pan" "潘 さん"
    python scripts/voice_library.py forget "参加者C"                                # ゴミ箱へ退避して外す
    python scripts/voice_library.py matches                                      # 候補を押した／断ったの集計

覚えるのは「名前の付いた人」だけ。自分（マイク側）と「不明話者」は覚えない。
会議のあとに自動で覚えるには `meeting.voice_library.enabled: true`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.audio.enrolled_diarizer import EnrolledDiarizer  # noqa: E402
from src.audio.voice_library import (  # noqa: E402
    MATCHES_FILE,
    VoiceLibrary,
    VoiceLibraryConfig,
    read_wav_mono,
    voices_from_session,
)

TRASH = REPO / "workspace" / "99-trash" / "voices"


def load_settings(path: Path | None) -> dict:
    for candidate in ([path] if path else []) + [REPO / "config" / "settings.yaml",
                                                  REPO / "config" / "settings.example.yaml"]:
        if candidate and candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return {}


def rows_of(session: Path) -> list[dict]:
    """作り直した全文があればそちらを使い、会議中の改名（renames.jsonl）を当てる。"""
    path = session / "transcripts_final.jsonl"
    if not path.exists():
        path = session / "transcripts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    renames_path = session / "renames.jsonl"
    if renames_path.exists():
        aliases = {}
        for line in renames_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                aliases[entry["old"]] = entry["new"]
        for row in rows:
            seen = set()
            while row.get("speaker") in aliases and row["speaker"] not in seen:
                seen.add(row["speaker"])
                row["speaker"] = aliases[row["speaker"]]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("sessions", type=Path, nargs="+")
    rename = sub.add_parser("rename")
    rename.add_argument("old")
    rename.add_argument("new")
    forget = sub.add_parser("forget")
    forget.add_argument("name")
    sub.add_parser("matches")
    args = parser.parse_args()

    settings = load_settings(args.config)
    meeting = settings.get("meeting") or {}
    config = VoiceLibraryConfig.from_mapping(meeting.get("voice_library"))
    path = Path(config.path).expanduser()
    path = path if path.is_absolute() else REPO / path
    library = VoiceLibrary(path, config)

    if args.command == "list":
        if not library.people:
            print(f"台帳は空です（{path}）")
        for name, entries in sorted(library.people.items()):
            sessions = "、".join(entry["session"] for entry in entries)
            print(f"  {name}: 会議 {len(entries)} 本（{sessions}）")
        return 0

    if args.command == "add":
        self_name = meeting.get("self_name", "自分")
        for session in args.sessions:
            recording = session / "recording_remote.wav"
            if not recording.exists():
                print(f"✗ {session.name}: 相手側の録音がありません")
                continue
            audio, sample_rate = read_wav_mono(recording)
            voices = voices_from_session(rows_of(session), audio, sample_rate, self_name=self_name,
                                         config=config, embed=EnrolledDiarizer.embed)
            result = library.learn(session.name, voices)
            for name, count in result.learned.items():
                print(f"  ✓ {session.name}: {name}（{count} 行）")
            for name, reason in result.skipped.items():
                print(f"  - {session.name}: {name} は覚えませんでした（{reason}）")
        library.save()
        return 0

    if args.command == "rename":
        library.rename(args.old, args.new)
        library.save()
        print(f"  {args.old} → {args.new}")
        return 0

    if args.command == "forget":
        moved = library.forget(args.name, TRASH)
        library.save()
        print(f"  {args.name} を台帳から外しました（退避先: {moved}）")
        return 0

    if args.command == "matches":
        entries = []
        for file in sorted((REPO / "workspace" / "sessions").glob(f"*/{MATCHES_FILE}")):
            entries += [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            print("まだ記録がありません（会議で候補を押す／断ると溜まります）")
            return 0
        # 「この類似度以上なら自動で付けてよいか」を決める材料。断られた最高点より上だけが安全圏
        accepted = sorted(e["score"] for e in entries if e["accepted"] and e.get("score") is not None)
        rejected = sorted(e["score"] for e in entries if not e["accepted"] and e.get("score") is not None)
        print(f"  押した {len(accepted)} 件: {accepted}")
        print(f"  断った {len(rejected)} 件: {rejected}")
        if rejected and accepted:
            safe = [score for score in accepted if score > max(rejected)]
            print(f"  断られた最高点 {max(rejected):.3f} より上で押された候補: {len(safe)}/{len(accepted)} 件")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
