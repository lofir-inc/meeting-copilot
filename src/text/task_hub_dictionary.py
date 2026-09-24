"""タスク管理 のクライアント辞書（Notion）と、この道具の置き換え辞書をつなぐ。

向きは 2 つ。

- **タスク管理 → ここ**: 確認済の対を手元にキャッシュし（`workspace/task_hub_dictionary.json`）、
  ①会議中と作り直しの置き換えに**名前の対だけ**使う ②辞書の候補づくりで「正しい表記」「既にある語」に使う
- **ここ → タスク管理**: 会議のあとの候補から人が選んだ対を、タスク管理 の辞書へ **`Status=候補`** で足す
  （候補は タスク管理 側でも自動適用されない。昇格は人が行う＝誤った学習が本文を書き換えない）

会議中に使うのは「正しい表記に英字を含む対」だけ（ITレビュー→Revuno・ネクシス→NEXIS）。
タスク管理 の辞書には文脈で決まる直し（「資料7の方」→「資料請求フォーム」）も確認済で入っていて、
会議の全行に当てると正しい発言を書き換える。そちらは従来どおり議事録づくり（/task_hub-minutes）で当たる。

Notion への読み書きは タスク管理 側の道具（`_shared/dict_correct.py`）に任せる。トークンの扱いを二重に持たない。
社内の設備なので、配布版では切ってある（`task_hub.dictionary_sync: false`）。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
CACHE_FILE = "workspace/task_hub_dictionary.json"
MIN_LIVE_LEN = 4
"""会議中に当てる誤りの最短の長さ（タスク管理 の自動適用と同じ。3 字以下は別の語に当たる）。"""

_FETCH = """
import json, sys
sys.path.insert(0, {shared!r})
import dict_correct as d
db = d.resolve_dictionary_db({client!r})
print(json.dumps({{"db": db, "pairs": d.load_pairs(db) if db else []}}, ensure_ascii=False))
"""


def fetch(shared_dir: str | Path, client: str, *, timeout_sec: float = 60.0) -> list[dict]:
    """タスク管理 の辞書を全件読む（候補・除外も含む。`status` 付き）。読めなければ例外。"""
    shared = str(Path(shared_dir).expanduser())
    if not (Path(shared) / "dict_correct.py").is_file():
        raise FileNotFoundError(f"タスク管理 の辞書の道具がありません: {shared}/dict_correct.py")
    result = subprocess.run(["python3", "-c", _FETCH.format(shared=shared, client=client)],
                            capture_output=True, text=True, timeout=timeout_sec, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-300:] or "タスク管理 の辞書を読めませんでした")
    payload = json.loads(result.stdout)
    if not payload.get("db"):
        raise RuntimeError(f"タスク管理 に {client} の辞書がありません")
    return list(payload.get("pairs") or [])


def save_cache(path: Path, pairs: list[dict], client: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"client": client, "at": datetime.now(JST).isoformat(timespec="seconds"),
                                "pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_cache(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("pairs") or [])
    except (json.JSONDecodeError, OSError):
        return []


def live_pairs(pairs: list[dict]) -> list[tuple[str, str]]:
    """会議中と作り直しで当ててよい対。確認済・4 字以上・略語でない・他の誤りの一部でない・正しい表記に英字。"""
    confirmed = [p for p in pairs if p.get("status") == "確認済" and p.get("wrong") and p.get("correct")]
    wrongs = [p["wrong"] for p in confirmed]
    picked: list[tuple[str, str]] = []
    for pair in confirmed:
        wrong, correct = pair["wrong"], pair["correct"]
        if len(wrong) < MIN_LIVE_LEN or re.fullmatch(r"[A-Za-z]{2,5}", wrong):
            continue
        if any(wrong != other and wrong in other for other in wrongs):
            continue
        if not re.search(r"[A-Za-z]", correct):
            continue            # 名前の対だけ。文脈で決まる直しは議事録づくりに任せる
        picked.append((wrong, correct))
    return picked


def known_wrongs(pairs: list[dict]) -> set[str]:
    """タスク管理 に既にある誤り（候補・除外も含む）。辞書の候補に同じものを出さない。"""
    return {p["wrong"] for p in pairs if p.get("wrong")}


def correct_notations(pairs: list[dict]) -> list[str]:
    """タスク管理 が知っている正しい表記（除外を除く）。"""
    return list(dict.fromkeys(p["correct"] for p in pairs if p.get("correct") and p.get("status") != "除外"))


def add_candidates(shared_dir: str | Path, client: str, pairs: list[tuple[str, str]], *,
                   timeout_sec: float = 60.0) -> list[dict]:
    """選ばれた対を タスク管理 の辞書へ **候補** として足す（1 件ずつ・結果を返す）。"""
    tool = Path(shared_dir).expanduser() / "dict_correct.py"
    if not tool.is_file():
        raise FileNotFoundError(f"タスク管理 の辞書の道具がありません: {tool}")
    results = []
    for wrong, correct in pairs:
        run = subprocess.run(["python3", str(tool), "--client", client, "--add", wrong, correct],
                             capture_output=True, text=True, timeout=timeout_sec, check=False)
        try:
            results.append(json.loads(run.stdout))
        except json.JSONDecodeError:
            results.append({"status": "error", "wrong": wrong, "detail": (run.stderr or run.stdout).strip()[-200:]})
    return results


def merge_pairs(primary: list[tuple[str, str]], extra: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """エンジンの辞書を優先して足し合わせ、長い語から並べる（短い語を先に当てると長い語が壊れる）。"""
    table = dict(extra)
    table.update(dict(primary))
    return sorted(table.items(), key=lambda pair: len(pair[0]), reverse=True)
