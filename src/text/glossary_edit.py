"""置き換え辞書を画面から直す（見る・足す・外す）。YAML を書き直さず、行で差し込む／抜く。

コメント（なぜその語を入れたか・実測の経緯）が辞書の価値の半分なので、`yaml.dump` で書き直さない。
外すときは、直す前の辞書を `workspace/99-trash/glossary/` に退避してから外す（直接消さない）。
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.text.glossary_candidates import append_to_glossary

JST = ZoneInfo("Asia/Tokyo")


def read_entries(path: Path) -> dict:
    """辞書の中身（置き換えと守り札）。無ければ空。"""
    path = Path(path)
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    table = data.get("replacements") or {}
    return {
        "replacements": [{"wrong": str(wrong), "right": str(right)} for wrong, right in table.items()],
        "protect": [str(word) for word in (data.get("protect") or [])],
    }


def add_replacement(path: Path, wrong: str, right: str, note: str) -> bool:
    """置き換えを 1 つ足す。既にある語は足さない（上書きしたいときは外してから足す）。"""
    wrong, right = wrong.strip(), right.strip()
    if not wrong or not right or wrong == right:
        raise ValueError("誤りと正しい表記の両方を、違う語で入れてください")
    _refuse_breaking_guard(path, wrong)
    added, _ = append_to_glossary(Path(path), [(wrong, right)], note)
    return bool(added)


def add_guard(path: Path, word: str, note: str) -> bool:
    word = word.strip()
    if not word:
        raise ValueError("守り札の語が空です")
    _, added = append_to_glossary(Path(path), [], note, [word])
    return bool(added)


def remove_replacement(path: Path, wrong: str, trash_dir: Path) -> Path:
    return _remove(Path(path), "replacements", wrong, trash_dir)


def remove_guard(path: Path, word: str, trash_dir: Path) -> Path:
    return _remove(Path(path), "protect", word, trash_dir)


def _refuse_breaking_guard(path: Path, wrong: str) -> None:
    """守り札そのものを誤りとして足すのは止める（守った語を壊す辞書になる）。"""
    if wrong in read_entries(path)["protect"]:
        raise ValueError(f"「{wrong}」は守り札に入っています（壊さない語として登録済み）")


def _remove(path: Path, section: str, word: str, trash_dir: Path) -> Path:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start, end = _section_bounds(lines, section)
    if start is None:
        raise KeyError(f"辞書に {section} がありません")
    target = None
    for index in range(start + 1, end):
        if _line_word(lines[index], section) == word:
            target = index
            break
    if target is None:
        raise KeyError(f"辞書にありません: {word}")
    backup = _backup(path, trash_dir)
    del lines[target]
    updated = "".join(lines)
    yaml.safe_load(updated)             # 壊れた YAML を書かない
    path.write_text(updated, encoding="utf-8")
    return backup


def _section_bounds(lines: list[str], section: str) -> tuple[int | None, int]:
    start = next((i for i, line in enumerate(lines) if re.match(rf"^{section}:", line)), None)
    if start is None:
        return None, len(lines)
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^[A-Za-z_][\w-]*:", lines[i])), len(lines))
    return start, end


def _line_word(line: str, section: str) -> str | None:
    """その行が表す語（置き換えならキー・守り札なら値）。コメント行や空行は None。"""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        parsed = yaml.safe_load(stripped)
    except yaml.YAMLError:
        return None
    if section == "replacements" and isinstance(parsed, dict) and len(parsed) == 1:
        return str(next(iter(parsed)))
    if section == "protect" and isinstance(parsed, list) and len(parsed) == 1:
        return str(parsed[0])
    return None


def _backup(path: Path, trash_dir: Path) -> Path:
    trash_dir = Path(trash_dir)
    trash_dir.mkdir(parents=True, exist_ok=True)
    target = trash_dir / f"{path.stem}-{datetime.now(JST):%Y%m%d-%H%M%S-%f}{path.suffix}"
    shutil.copy2(path, target)
    return target


def record_decision(session_dir: Path, *, kind: str, wrong: str, right: str, accepted: bool) -> None:
    """候補を入れた／見送ったを残す（次に候補を作り直したとき、同じものを出さない）。"""
    entry = {"at": datetime.now(JST).isoformat(timespec="seconds"), "kind": kind, "wrong": wrong,
             "right": right, "accepted": accepted}
    with (Path(session_dir) / DECISIONS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def decided(session_dir: Path) -> set[tuple[str, str]]:
    """もう決めた候補（(kind, wrong)）。"""
    path = Path(session_dir) / DECISIONS_FILE
    if not path.exists():
        return set()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            found.add((entry.get("kind", "replace"), entry.get("wrong", "")))
    return found


DECISIONS_FILE = "glossary_decisions.jsonl"
