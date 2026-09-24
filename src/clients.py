"""この会議は**どのクライアントの会議か**を選ぶ。議事録・タスク・辞書の行き先がこれで決まる。

2026-09-15 の事故: ミナトの定例（案件 0045）の議事録が **自社 の議事録DB**に登録された。
設定の `task_hub.client_name: 自社` が固定で使われ、**会議ごとに選ぶ段がどこにも無かった**。
タスク 18 行も同じ理由で 自社 側へ入った。

決め方（運用者 指示 2026-09-16）:

- 会議の**前**に、登録済みのクライアントから選ぶ（画面。Notion の Client Master の一覧をそのまま出す）
- 選ばなければ **自社（自社）**。意見交換どまり・案件化していない相手は自社でよい
- **会議の終わりにもう一度聞く**（議事録を引き渡す前）。初回面談がその場で案件になることがあるため
- 選んだクライアントは、議事録とタスクの行き先だけでなく、**会議中の辞書**（そのクライアントの
  Notion 辞書）にも効く。今後、過去の発言・製品 Fact を会議中に引く土台にする

一覧は手元に控える（`workspace/clients.json`）。Notion に繋がらない日でも会議は始められる。
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
CACHE_FILE = "workspace/clients.json"
CLIENT_FILE = "client.json"
"""セッションに残す「この会議のクライアント」。議事録の引き渡しはこれを見る。"""

SELF_CLIENT = "自社"
"""選ばなかったときの行き先（自社）。"""

_FETCH = """
import json, sys
sys.path.insert(0, {shared!r})
import task_hub_context as C, task_hub_notion as N
out, by_page = [], {{}}
for row in N.query_database(C.CLIENT_MASTER_DB):
    props = row["properties"]
    entry = {{"name": N.prop_title(props, "client_name"),
             "client_id": N.prop_text(props, "client_id"),
             "minutes_db_id": N.prop_text(props, "minutes_db_id"),
             "people": []}}
    out.append(entry)
    by_page[row.get("id")] = entry

# People Master のその会社の人（名前＋話者ラベル）を控えへ。
# 議事録のたびに Notion を叩かないための先取り。読めなければ people は空のまま。
people_db = {people_db!r}
if people_db:
    try:
        for row in N.query_database(people_db):
            props = row["properties"]
            values = [N.prop_title(props)] + list(N.prop_multi_select(props, "Speaker Labels"))
            for link in (props.get("Organization", {{}}) or {{}}).get("relation", []):
                entry = by_page.get(link.get("id"))
                if entry is None:
                    continue
                for value in values:
                    value = (value or "").strip()
                    if value and value not in entry["people"]:
                        entry["people"].append(value)
    except Exception as error:
        print("people-master-failed: %s" % error, file=sys.stderr)
print(json.dumps(out, ensure_ascii=False))
"""


def fetch(shared_dir: str | Path, *, people_db: str = "", timeout_sec: float = 60.0) -> list[dict]:
    """Client Master の一覧を読む（名前・client_id・議事録DB の有無・**その会社の人**）。

    `people_db`（People Master）を渡すと、会社ごとの人（名前＋話者ラベル）も控える。
    議事録を作るときは手元の控えを読むだけにしたい（仕上げの経路に Notion を挟まない）。
    People Master が読めなくても、クライアント一覧は返す（people が空になるだけ）。
    """
    shared = str(Path(shared_dir).expanduser())
    if not (Path(shared) / "task_hub_context.py").is_file():
        raise FileNotFoundError(f"タスク管理 の共有モジュールがありません: {shared}")
    result = subprocess.run(["python3", "-c", _FETCH.format(shared=shared, people_db=people_db)],
                            capture_output=True, text=True, timeout=timeout_sec, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-300:] or "クライアント一覧を読めませんでした")
    return [entry for entry in json.loads(result.stdout) if entry.get("name")]


def save_cache(path: Path, clients: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": datetime.now(JST).isoformat(timespec="seconds"), "clients": clients},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def load_cache(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("clients") or [])
    except (json.JSONDecodeError, OSError):
        return []


def people_of(clients: list[dict], name: str) -> list[str]:
    """その会社の人（People Master 由来）。見つからなければ空。"""
    for entry in clients:
        if str(entry.get("name", "")).strip() == str(name).strip():
            return [str(one) for one in (entry.get("people") or []) if str(one).strip()]
    return []


def options(clients: list[dict]) -> list[dict]:
    """画面に出す並び。自社を先頭に、実在のクライアント、最後にテスト用。"""
    def rank(entry: dict) -> tuple[int, str]:
        name, client_id = entry.get("name", ""), str(entry.get("client_id", ""))
        if name == SELF_CLIENT:
            return (0, name)
        if not client_id or _looks_like_test(name, client_id):
            return (2, name)
        return (1, name)

    return [{"name": entry["name"], "client_id": str(entry.get("client_id", "")),
             "has_minutes_db": bool(entry.get("minutes_db_id")),
             "test": _looks_like_test(entry.get("name", ""), str(entry.get("client_id", "")))}
            for entry in sorted(clients, key=rank)]


def _looks_like_test(name: str, client_id: str) -> bool:
    lowered = name.lower()
    return (not client_id or client_id in {"sample", "9998", "9999"}
            or any(mark in lowered for mark in ("sample", "テスト", "検証")))


def read(session_dir: Path) -> dict | None:
    """この会議で選ばれたクライアント（まだなら None）。"""
    path = Path(session_dir) / CLIENT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def choose(session_dir: Path, name: str, *, client_id: str = "", how: str = "画面") -> dict:
    """この会議のクライアントを決めて残す。空の名前は受け付けない（黙って自社に倒さない）。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("クライアント名が空です")
    entry = {"name": name, "client_id": client_id,
             "at": datetime.now(JST).isoformat(timespec="seconds"), "how": how}
    path = Path(session_dir) / CLIENT_FILE
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("この会議のクライアント: %s（%s）", name, how)
    return entry


def effective(session_dir: Path, default: str = SELF_CLIENT) -> tuple[str, bool]:
    """(行き先のクライアント名, 人が選んだか)。選んでいなければ自社（設定のクライアント名）へ倒す。"""
    chosen = read(session_dir)
    if chosen and chosen.get("name"):
        return str(chosen["name"]), True
    return default or SELF_CLIENT, False
