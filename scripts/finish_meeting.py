#!/usr/bin/env python3
"""会議のあとの仕上げを、**あとから**走らせる（作り直し・声の台帳・辞書の候補・議事録）。

    python scripts/finish_meeting.py workspace/sessions/<会議>
    python scripts/finish_meeting.py workspace/sessions/<会議> --only minutes

会議の終わりに「あとでやる」を選んだとき、ブラウザを閉じたあとでもここから再開できる。
  画面からやるなら「会議アシスタント.command」→「会議のあと」→「仕上げる」（中で同じものを呼ぶ）。
もう終わっている工程は飛ばす（`transcripts_final.jsonl` があれば作り直しはしない）。
途中で失敗しても、そこまでの成果物は残る。もう一度実行すれば続きから進む。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src import clients, finish  # noqa: E402
from src.audio import ffmpeg  # noqa: E402

PYTHON = sys.executable


def load_settings(path: Path | None) -> dict:
    for candidate in ([path] if path else []) + [REPO / "config" / "settings.yaml",
                                                  REPO / "config" / "settings.example.yaml"]:
        if candidate and candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return {}


def voice_library_path(settings: dict) -> Path:
    """声の台帳の場所（画面の `_voice_path` と同じ決め方）。"""
    config = (settings.get("meeting") or {}).get("voice_library") or {}
    path = Path(str(config.get("path", "workspace/voices/library.json"))).expanduser()
    return path if path.is_absolute() else REPO / path


def run_finalize(session: Path, settings: dict) -> bool:
    """録音から全文を作り直す（外へ出すかは finalize_meeting.py 側の関門で聞かれる）。"""
    if not (session / "recording_remote.wav").exists():
        print("  録音がないので作り直しは飛ばします")
        return True
    argv = [PYTHON, str(REPO / "scripts" / "finalize_meeting.py"), str(session)]
    return subprocess.run(argv, check=False).returncode == 0


def learn_voices(session: Path, settings: dict) -> bool:
    """名前の付いた人の声を台帳に覚える（`scripts/voice_library.py add` と同じ）。"""
    meeting = settings.get("meeting") or {}
    if not (meeting.get("voice_library") or {}).get("enabled"):
        return True
    argv = [PYTHON, str(REPO / "scripts" / "voice_library.py"), "add", str(session)]
    return subprocess.run(argv, check=False).returncode == 0


def build_candidates(session: Path, settings: dict) -> bool:
    from glossary_candidates import refresh          # noqa: PLC0415 — スクリプト間の再利用

    path, candidates, _ = refresh(session, settings)
    print(f"  辞書の候補 {len(candidates)} 件: {path.name}")
    return True


def make_minutes(session: Path, settings: dict) -> bool:
    """議事録を作る。外（Gemini）へ出すかは、その会議で承認したかで決まる。"""
    argv = [PYTHON, str(REPO / "scripts" / "make_minutes.py"), str(session)]
    sends = session / "external_sends.jsonl"
    approved = sends.exists() and any('"note": "live_batch"' in line or '"note": "state"' in line
                                      for line in sends.read_text(encoding="utf-8").splitlines())
    argv.append("--approved-external" if approved else "--no-external")
    return subprocess.run(argv, check=False).returncode == 0


def compress_audio(session: Path, settings: dict) -> bool:
    """録音を mp3 に畳む（マイク別＋統合した 1 本）。元の wav は残す。"""
    from src.audio.archive import ArchiveConfig, compress, wav_bytes

    config = ArchiveConfig.from_mapping((settings.get("meeting") or {}).get("audio_archive"))
    if not config.enabled:
        return True
    before = wav_bytes(session)
    made = compress(session, config)
    if not made:
        print("  畳む録音がありません")
        return True
    total = sum(made.values())
    print("  " + "／".join(f"{key}.mp3 {size / 1048576:.0f} MB" for key, size in made.items()))
    if before:
        print(f"  元の wav {before / 1048576:.0f} MB → mp3 {total / 1048576:.0f} MB"
              f"（{before / total:.0f} 分の 1）。wav は残してあります")
    return True


RUNNERS = {"finalize": run_finalize, "voices": learn_voices,
           "glossary": build_candidates, "minutes": make_minutes, "audio": compress_audio}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--only", choices=list(RUNNERS), action="append",
                        help="この工程だけ走らせる（繰り返し指定できる）")
    parser.add_argument("--client", help="この会議の相手（議事録とタスクの行き先）")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--drop-wav", action="store_true",
                        help="mp3 に畳んだあと、元の wav をゴミ箱へ退避する（作り直しはできなくなる）")
    args = parser.parse_args()
    session = args.session.resolve()
    if not session.is_dir():
        print(f"✗ その会議はありません: {session}")
        return 1
    # ここで止める。無いまま進むと、作り直しが生のトレースバックで落ち、
    #   そのあと「議事録の材料がありません」と連鎖して、本当の原因が最後まで出てこない。
    ffmpeg.require("会議のあとの仕上げ")
    settings = load_settings(args.config)
    if args.client:
        clients.choose(session, args.client, how="あとからの仕上げ")

    name, chosen = clients.effective(session, ((settings.get("task_hub") or {}).get("client_name", clients.SELF_CLIENT)))
    print(f"会議: {session.name}／相手: {name}{'' if chosen else '（未選択なので自社）'}")

    # 声の台帳を使わない設定なら、その工程は最初から終わり扱い（さもないと毎回残り続ける）
    voice_enabled = bool(((settings.get("meeting") or {}).get("voice_library") or {}).get("enabled"))
    # 台帳の場所も渡す（画面と同じ判定）。渡さないと声を覚え終えても「残り」に出続け、
    #   pending.json をまた書いてしまう（2026-09-24 に発見）
    voice_library = voice_library_path(settings)
    steps = finish.remaining(session, wanted=args.only, voice_library=voice_library,
                             voice_enabled=voice_enabled)
    if not steps:
        print("  残っている仕上げはありません")
        finish.clear_pending(session)
        return 0

    failed = []
    for step in steps:
        print(f"\n── {step.label}（{step.minutes}）")
        try:
            if not RUNNERS[step.key](session, settings):
                failed.append(step)
        except Exception as error:  # noqa: BLE001 — 1 つ失敗しても残りは進める
            print(f"  ✗ {step.label}: {error}")
            failed.append(step)

    left = finish.remaining(session, voice_library=voice_library, voice_enabled=voice_enabled)
    if left:
        finish.write_pending(session, left, note="あとからの仕上げで残ったぶん")
        print(f"\n  残り: {'／'.join(step.label for step in left)}（もう一度実行すれば続きから進みます）")
    else:
        finish.clear_pending(session)
        print("\n  仕上げが終わりました")
    if args.drop_wav:
        from src.audio.archive import drop_wav

        try:
            moved = drop_wav(session, REPO / "workspace" / "99-trash")
            print(f"  元の wav {len(moved)} 本をゴミ箱へ退避しました")
        except RuntimeError as error:
            print(f"  wav は退避しませんでした（{error}）")
    handoff = session / "minutes_handoff.json"
    if handoff.exists():
        print(f"  議事録の材料: {json.loads(handoff.read_text(encoding='utf-8')).get('transcript_path', handoff)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
