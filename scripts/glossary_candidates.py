#!/usr/bin/env python3
"""会議のあとに、置き換え辞書の候補を出す／チェックしたものだけ辞書に足す。手元だけで動く。

    python scripts/glossary_candidates.py workspace/sessions/<会議>            # 候補を作り直して表示
    python scripts/glossary_candidates.py workspace/sessions/<会議> --accept   # チェックしたものを辞書へ

会議の終わり（と議事録を作ったあと）に自動で `glossary_candidates.md` ができる。
そのファイルで辞書に入れたい行の `[ ]` を `[x]` にしてから `--accept` を実行する。

足す先は、その会議の文字起こしに使ったエンジンの辞書（Whisper → config/glossary.yaml、
  外の Gemini → config/glossary-gemini.yaml）。エンジンごとに誤り方が違うため。
足した語は次の会議から効く（会議中の置き換えと、会議のあとの作り直しの両方）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.text.glossary_candidates import (  # noqa: E402
    CANDIDATES_FILE,
    append_to_glossary,
    build_for_session,
    checked,
    glossary_for_session,
)


def load_settings(path: Path | None) -> dict:
    for candidate in ([path] if path else []) + [REPO / "config" / "settings.yaml",
                                                  REPO / "config" / "settings.example.yaml"]:
        if candidate and candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return {}


def library_names(meeting: dict) -> list[str]:
    """声の台帳にいる人の名前（正しい表記の手がかり）。台帳を使っていなければ空。"""
    config = meeting.get("voice_library") or {}
    if not config.get("enabled"):
        return []
    from src.audio.voice_library import VoiceLibrary

    path = Path(config.get("path", "workspace/voices/library.json")).expanduser()
    path = path if path.is_absolute() else REPO / path
    try:
        return list(VoiceLibrary(path).people)
    except (OSError, ValueError):
        return []


def task_hub_pairs(settings: dict) -> list[dict]:
    """タスク管理 の辞書のキャッシュ（つないでいなければ空）。"""
    if not (settings.get("task_hub") or {}).get("dictionary_sync"):
        return []
    from src.text import task_hub_dictionary

    return task_hub_dictionary.load_cache(REPO / task_hub_dictionary.CACHE_FILE)


def push_to_task_hub(settings: dict, pairs: list[tuple[str, str]], session: str) -> None:
    """選んだ対を タスク管理 の辞書へ **候補** として足す（自動適用されない。昇格は タスク管理 側で人が行う）。"""
    task_hub = settings.get("task_hub") or {}
    if not task_hub.get("dictionary_sync") or not pairs:
        return
    from src.text import task_hub_dictionary

    try:
        results = task_hub_dictionary.add_candidates(task_hub.get("shared_dir", "~/.claude/skills/_shared"),
                                                  task_hub.get("client_name", "自社"), pairs)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  タスク管理 の辞書には足せませんでした（{error}）")
        return
    for result in results:
        status = {"added": "候補として追加", "already_exists": "既にある"}.get(result.get("status"), result.get("status"))
        print(f"  タスク管理: 「{result.get('wrong')}」 {status}{'（' + result['detail'] + '）' if result.get('detail') else ''}")


def refresh(session: Path, settings: dict) -> tuple[Path, list, Path]:
    meeting = settings.get("meeting") or {}
    glossary = glossary_for_session(session, meeting, REPO)
    path, candidates = build_for_session(session, glossary_path=glossary,
                                         self_name=meeting.get("self_name", "自分"),
                                         extra_names=library_names(meeting),
                                         task_hub_pairs=task_hub_pairs(settings))
    return path, candidates, glossary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--accept", action="store_true", help="チェックした候補を辞書に足す")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config)

    if args.accept:
        path = args.session / CANDIDATES_FILE
        if not path.exists():
            print(f"✗ 候補のファイルがありません: {path}")
            return 1
        glossary = glossary_for_session(args.session, settings.get("meeting") or {}, REPO)
        pairs, guards = checked(path)
        if not pairs and not guards:
            print(f"チェックの付いた候補がありません（{path} の [ ] を [x] にしてください）")
            return 0
        added, added_guards = append_to_glossary(glossary, pairs, args.session.name, guards)
        for wrong, right in added:
            print(f"  ✓ 置き換え 「{wrong}」→「{right}」")
        for word in added_guards:
            print(f"  ✓ 守り札 「{word}」")
        skipped = len(pairs) + len(guards) - len(added) - len(added_guards)
        if skipped:
            print(f"  - {skipped} 件は辞書に既にありました")
        print(f"  → {glossary}（次の会議から効きます）")
        push_to_task_hub(settings, added, args.session.name)
        return 0

    path, candidates, glossary = refresh(args.session, settings)
    print(f"候補 {len(candidates)} 件 → {path}")
    for candidate in candidates[:20]:
        if candidate.kind == "protect":
            print(f"  守り札「{candidate.wrong}」（{candidate.count} 回・{candidate.source}）{candidate.reason}")
        else:
            print(f"  「{candidate.wrong}」→「{candidate.right}」（{candidate.count} 回・{candidate.source}）")
    print(f"  足す先: {glossary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
