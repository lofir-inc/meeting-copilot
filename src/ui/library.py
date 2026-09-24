"""画面「会議アシスタント」— 辞書の候補を選ぶ・辞書を直す・声の台帳を直す。ファイルを手で編集しない。

会議中はダッシュボード（`/library`）から、会議のあとは `scripts/library_ui.py` で同じ画面が開く。
会議中に辞書を直したら、その場で動いている会議にも効かせる（`on_change`）。

2026-09-15 運用者 依頼「辞書なども markdown ファイルの編集ではなく、GUI でそのまま入力できて欲しい」。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from datetime import datetime
from zoneinfo import ZoneInfo

from src import capability, clients, finish, settings_edit
from src.app import meeting_launch
from src.audio import archive
from src.screen import capture as screen_capture
from src.text import glossary_edit
from src import task_hub_minutes
from src.llm.claude_cli_client import ClaudeCliClient
from src.text.glossary_candidates import CANDIDATES_JSON, Candidate, glossary_for_session, write_candidates

SESSIONS_SHOWN = 20
EDITS_FILE = "text_edits.jsonl"
"""聞き直して直した記録（元の文も残す）。議事録を作る前に直すための窓口。"""
JST = ZoneInfo("Asia/Tokyo")
logger = logging.getLogger(__name__)


def _screens(session_dir: Path) -> list[dict]:
    """会議中に控えた画面（発言の隣に出すので、時刻つきで返す）。"""
    path = session_dir / screen_capture.SCREENS_FILE
    if not path.exists():
        return []
    found = []
    for entry in _read_lines(path):
        found.append({"at": float(entry.get("at", 0)), "file": str(entry.get("file", "")),
                      "title": str(entry.get("title", "")), "urls": list(entry.get("urls") or [])})
    return found


def _participants(rows: list[dict]) -> list[dict]:
    """誰がどれだけ話したか（会議中の画面の「参加者」と同じ役割）。"""
    found: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("speaker", ""))
        if not name:
            continue
        entry = found.setdefault(name, {"name": name, "lines": 0, "seconds": 0.0})
        entry["lines"] += 1
        entry["seconds"] += max(0.0, float(row.get("end_time", 0)) - float(row.get("start_time", 0)))
    return sorted(({**entry, "minutes": round(entry["seconds"] / 60)} for entry in found.values()),
                  key=lambda entry: entry["seconds"], reverse=True)


def _sections(state: dict) -> list[dict]:
    """論点（話題）ごとに、決定事項・TODO・質問をまとめる。

    決定事項は論点と直接は結ばれていないので、**話題が切り替わった時刻で区切る**
    （運用者 指摘 2026-09-16「29 個並べられても読む気が起きない。何についての決定か分からない」）。
    """
    topics = sorted((_item(value) | {"created_at": float(value.get("created_at", 0)),
                                     "status": str(value.get("status", ""))}
                     for value in state.get("topics") or [] if isinstance(value, dict)),
                    key=lambda topic: topic["created_at"])
    buckets = [{"topic": topic["text"], "status": topic["status"], "at": topic["created_at"],
                "decisions": [], "todos": [], "questions": []} for topic in topics]
    other = {"topic": "（話題の前・どこにも入らないもの）", "status": "", "at": 0.0,
             "decisions": [], "todos": [], "questions": []}

    def place(kind: str, value: dict) -> None:
        at = float(value.get("created_at", 0))
        target = other
        for bucket in buckets:
            if at >= bucket["at"]:
                target = bucket
            else:
                break
        target[kind].append(_item(value))

    for kind in ("decisions", "todos", "questions"):
        for value in state.get(kind) or []:
            if isinstance(value, dict):
                place(kind, value)
    sections = ([other] if any(other[kind] for kind in ("decisions", "todos", "questions")) else []) + buckets
    return [section for section in sections
            if section["decisions"] or section["todos"] or section["questions"]]


def _item(value) -> dict:
    """状態の 1 件（決定事項・TODO 等）を画面用に薄くする。"""
    if isinstance(value, dict):
        return {"text": str(value.get("text", "")), "owner": str(value.get("owner") or value.get("by") or ""),
                "status": str(value.get("status", "")), "due": str(value.get("due", "")),
                "note": str(value.get("note", ""))}
    return {"text": str(value), "owner": "", "status": "", "due": "", "note": ""}


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return found


def _rows(session_dir: Path) -> list[dict]:
    """その会議の全文（作り直したものがあればそちら）。"""
    path = session_dir / "transcripts_final.jsonl"
    if not path.exists():
        path = session_dir / "transcripts.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Library:
    """画面から呼ばれる中身（テストしやすいように FastAPI から切り離す）。"""

    def __init__(
        self,
        repo: Path,
        settings: Callable[[], dict],
        *,
        sessions_dir: Path | None = None,
        on_change: Callable[[str], None] | None = None,
        current_dictionary: Callable[[], str | None] | None = None,
        push_to_task_hub: Callable[[list[tuple[str, str]], str], list[dict]] | None = None,
    ) -> None:
        self.repo = Path(repo)
        self._settings = settings
        self.sessions_dir = Path(sessions_dir) if sessions_dir else self.repo / "workspace" / "sessions"
        self.trash = self.repo / "workspace" / "99-trash"
        self._on_change = on_change or (lambda kind: None)
        self._current = current_dictionary or (lambda: None)
        self._push = push_to_task_hub
        self._running: dict[str, subprocess.Popen] = {}
        """仕上げを走らせている会議（画面から押したぶん）。"""

    # ------------------------------------------------------------------ 辞書

    def dictionaries(self) -> list[dict]:
        meeting = self._settings().get("meeting") or {}
        found = [("whisper", "手元の文字起こし（Whisper）用", meeting.get("glossary_path", "config/glossary.yaml"))]
        external = (meeting.get("external_stt") or {}).get("glossary_path")
        if external:
            found.append(("gemini", "外の文字起こし（Gemini）用", external))
        current = self._current()
        if current is None:
            # 会議の外では、設定で使っている文字起こしのエンジンの辞書を先に出す
            current = "gemini" if str((self._settings().get("stt") or {}).get("engine", "")).lower() == "gemini" and external else "whisper"
        return [{"id": key, "label": label, "path": str(self._path(value)), "current": key == current,
                 **glossary_edit.read_entries(self._path(value))} for key, label, value in found]

    def dictionary_path(self, dictionary: str) -> Path:
        entries = self.dictionaries()
        if dictionary in {"", "current"}:
            dictionary = next((entry["id"] for entry in entries if entry["current"]), "whisper")
        for entry in entries:
            if entry["id"] == dictionary:
                return Path(entry["path"])
        raise KeyError(f"その辞書はありません: {dictionary}")

    def add(self, dictionary: str, wrong: str, right: str) -> dict:
        path = self.dictionary_path(dictionary)
        added = glossary_edit.add_replacement(path, wrong, right, "画面から足した")
        if added:
            self._on_change("glossary")
        task_hub = self._push([(wrong.strip(), right.strip())], "画面") if added and self._push else []
        return {"ok": True, "added": added, "task_hub": task_hub}

    def remove(self, dictionary: str, wrong: str) -> dict:
        backup = glossary_edit.remove_replacement(self.dictionary_path(dictionary), wrong, self.trash / "glossary")
        self._on_change("glossary")
        return {"ok": True, "backup": str(backup)}

    def add_guard(self, dictionary: str, word: str) -> dict:
        added = glossary_edit.add_guard(self.dictionary_path(dictionary), word, "画面から足した")
        if added:
            self._on_change("glossary")
        return {"ok": True, "added": added}

    def remove_guard(self, dictionary: str, word: str) -> dict:
        backup = glossary_edit.remove_guard(self.dictionary_path(dictionary), word, self.trash / "glossary")
        self._on_change("glossary")
        return {"ok": True, "backup": str(backup)}

    # ------------------------------------------------------------------ 候補

    def sessions(self) -> list[dict]:
        found = []
        for path in sorted(self.sessions_dir.glob(f"*/{CANDIDATES_JSON}"), reverse=True)[:SESSIONS_SHOWN]:
            found.append({"name": path.parent.name, "pending": len(self._read_candidates(path.parent))})
        return found

    def candidates(self, session: str) -> dict:
        session_dir = self._session(session)
        target = glossary_for_session(session_dir, self._settings().get("meeting") or {}, self.repo)
        dictionary = next((entry["id"] for entry in self.dictionaries() if Path(entry["path"]) == target), "whisper")
        return {"session": session, "dictionary": dictionary, "items": self._read_candidates(session_dir)}

    def decide(self, session: str, items: list[dict]) -> dict:
        """選んだ候補を辞書へ入れる／見送る。入れた置き換えは タスク管理 にも「候補」で送る（つないでいれば）。"""
        session_dir = self._session(session)
        target = self.dictionary_path(self.candidates(session)["dictionary"])
        results, pushed = [], []
        for item in items:
            kind = item.get("kind", "replace")
            wrong = str(item.get("wrong", "")).strip()
            right = str(item.get("right", "")).strip()
            accepted = bool(item.get("accept"))
            outcome = "見送り"
            if accepted:
                try:
                    if kind == "protect":
                        added = glossary_edit.add_guard(target, wrong, session)
                    else:
                        added = glossary_edit.add_replacement(target, wrong, right, session)
                        if added:
                            pushed.append((wrong, right))
                    outcome = "入れました" if added else "辞書に既にありました"
                except ValueError as error:
                    results.append({"wrong": wrong, "outcome": f"入れられません: {error}"})
                    continue
            glossary_edit.record_decision(session_dir, kind=kind, wrong=wrong, right=right, accepted=accepted)
            results.append({"wrong": wrong, "outcome": outcome})
        remaining = self._read_candidates(session_dir)
        write_candidates(session_dir, [Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
                                       for c in remaining], target)
        (session_dir / CANDIDATES_JSON).write_text(json.dumps(remaining, ensure_ascii=False, indent=1), encoding="utf-8")
        if any(r["outcome"] == "入れました" for r in results):
            self._on_change("glossary")
        task_hub = self._push(pushed, session) if pushed and self._push else []
        return {"ok": True, "results": results, "task_hub": task_hub, "pending": len(remaining)}

    def _read_candidates(self, session_dir: Path) -> list[dict]:
        path = session_dir / CANDIDATES_JSON
        if not path.exists():
            return []
        done = glossary_edit.decided(session_dir)
        return [item for item in json.loads(path.read_text(encoding="utf-8"))
                if (item.get("kind", "replace"), item.get("wrong")) not in done]

    def _session(self, name: str) -> Path:
        path = (self.sessions_dir / name).resolve()
        if path.parent != self.sessions_dir.resolve() or not path.is_dir():
            raise KeyError(f"その会議はありません: {name}")
        return path

    # --------------------------------------------- 会議のあと（仕上げの再開）

    # ------------------------------------------------- ここから会議を始める

    MEETING_PORT = 8765
    MEETING_PORTS = 10
    """会議の画面を探す範囲。塞がっていたら会議は隣のポートで開くので、そこも見る（2026-09-17）。"""

    def meeting_status(self) -> dict:
        """会議の画面が動いているか（動いていればそちらへ案内する）。

        ポートが空いているかだけでは駄目（別の道具が同じ番号を使っていて「会議中です」と出た・2026-09-16）。
        会議の画面の口（`/api/status`）に聞いて、会議の名前が返るときだけ「会議中」と言う。
        """
        for offset in range(self.MEETING_PORTS):
            url = f"http://127.0.0.1:{self.MEETING_PORT + offset}/"
            payload = self._ask_status(url)
            if payload.get("session_name"):
                return {"running": True, "url": url, "session": payload["session_name"]}
        return {"running": False, "url": f"http://127.0.0.1:{self.MEETING_PORT}/"}

    @staticmethod
    def _ask_status(url: str) -> dict:
        """その口が会議の画面かを聞く。会議でなければ空を返す（例外にしない）。"""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{url}api/status", timeout=0.4) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def start_meeting(self) -> dict:
        """会議を始める。ターミナルは出さない。

        画面から始められるようにする（運用者 依頼 2026-09-16「立ち上げたら過去の振り返りも見たい」）。
        2026-09-18 に端末をやめた（運用者 指摘「ターミナルは一切見えないようになるといいね」）。
          会議中に入力を待つ 3 か所は、どれも**画面が無いときだけ**端末を使う作りで、
          話者登録は `meeting.skip_enrollment` で飛ばしている。∴ 端末は一度も要らない。
        立ち上がらなかったときのために、出力は `workspace/last_meeting.log` に落とす
          （切り離すと画面にも端末にも何も出ない）。
        """
        status = self.meeting_status()
        if status["running"]:
            return {"ok": True, "already": True, **status}
        meeting_launch.start(self.repo, Path(sys.executable))
        return {"ok": True, "started": True, **status}

    def clients(self) -> list[dict]:
        """登録済みクライアントの一覧（手元の控え）。画面の「相手」の選択肢。"""
        return clients.options(clients.load_cache(self.repo / clients.CACHE_FILE))

    def unfinished(self) -> list[dict]:
        """仕上げが残っている会議（新しい順）。ブラウザを閉じたあとでもここから再開する。"""
        found = []
        for path in sorted(self.sessions_dir.glob("*/minutes_input.md"), reverse=True)[:SESSIONS_SHOWN]:
            session_dir = path.parent
            steps = finish.remaining(session_dir, voice_library=self._voice_path(),
                                      voice_enabled=self._voice_enabled())
            if not steps and finish.read_pending(session_dir) is None:
                continue
            chosen = clients.read(session_dir)
            found.append({
                "name": session_dir.name,
                "steps": [{"key": s.key, "label": s.label, "minutes": s.minutes} for s in steps],
                "client": chosen.get("name") if chosen else None,
                "running": self._is_running(session_dir.name),
                "saved": finish.saved_already(session_dir),
            })
        return found

    def delete_meetings(self, sessions: list[str]) -> dict:
        """会議のログを消す。直接は消さず、ゴミ箱へ**フォルダごと退避**する（戻せる）。"""
        if not sessions:
            raise ValueError("消す会議が選ばれていません")
        trash = self.trash / f"{datetime.now(JST):%Y-%m-%d}_sessions"
        trash.mkdir(parents=True, exist_ok=True)
        moved = []
        for name in sessions:
            session_dir = self._session(name)
            target = trash / name
            if target.exists():
                target = trash / f"{name}-{datetime.now(JST):%H%M%S}"
            shutil.move(str(session_dir), str(target))
            moved.append(name)
        (trash / "WHY.md").write_text(
            "# なぜここにあるか\n\n画面「会議アシスタント」の会議の一覧から消した会議です"
            "（直接は消さず退避）。戻すには workspace/sessions/ へ移動し直してください。\n\n"
            + "\n".join(f"- {name}" for name in moved) + "\n", encoding="utf-8")
        return {"ok": True, "moved": moved, "trash": str(trash)}

    def set_client(self, session: str, client: str) -> dict:
        """一覧から、その会議の相手を決める（議事録とタスクの行き先）。空なら自社に戻す。"""
        session_dir = self._session(session)
        if not client.strip():
            path = session_dir / clients.CLIENT_FILE
            if path.exists():
                path.unlink()
            return {"ok": True, "client": None}
        known = {option["name"]: option["client_id"] for option in self.clients()}
        entry = clients.choose(session_dir, client, client_id=known.get(client, ""), how="会議の一覧")
        return {"ok": True, "client": entry["name"]}

    def finish_session(self, session: str, client: str = "") -> dict:
        """仕上げを走らせる（裏で回し、進み具合は `finish.log` に出る）。二重に走らせない。"""
        session_dir = self._session(session)
        if self._is_running(session):
            return {"ok": True, "already": True}
        if client:
            clients.choose(session_dir, client, how="会議のあとの画面")
        script = self.repo / "scripts" / "finish_meeting.py"
        log = (session_dir / finish.LOG_FILE).open("a", encoding="utf-8")
        log.write(f"\n=== 仕上げを始めます\n")
        log.flush()
        self._running[session] = subprocess.Popen(
            [sys.executable, str(script), str(session_dir)],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        return {"ok": True, "started": True}

    def finish_status(self, session: str) -> dict:
        """仕上げの進み具合（走っているか・ログの終わり・残り）。"""
        session_dir = self._session(session)
        log = session_dir / finish.LOG_FILE
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:] if log.exists() else []
        # 失敗して終わったことを画面へ返す（2026-09-24: ffmpeg が見つからず 1 秒で落ちたのに、
        #   一覧は読み直すだけで何も出ず「押しても変わらない」に見えた）
        process = self._running.get(session)
        failed = process is not None and process.poll() not in (None, 0)
        marked = [line for line in tail if line.startswith("✗")]
        error = (marked[-1] if marked else next((line for line in reversed(tail) if line.strip()), "")) if failed else ""
        return {"session": session, "running": self._is_running(session), "lines": tail,
                "failed": failed, "error": error,
                "steps": [{"key": s.key, "label": s.label}
                          for s in finish.remaining(session_dir, voice_library=self._voice_path(),
                                      voice_enabled=self._voice_enabled())]}

    def _is_running(self, session: str) -> bool:
        process = self._running.get(session)
        return process is not None and process.poll() is None

    # ------------------------------------------- 過去の会議（一覧・聞き直す・直す）

    def meetings(self, limit: int = 30) -> list[dict]:
        """終わった会議の一覧（新しい順）。あとから振り返る入口。"""
        found = []
        # 名前順ではなく**新しい順**（テスト用の会議名が上に来ないように）
        paths = sorted(self.sessions_dir.glob("*/transcripts.jsonl"),
                       key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths[:limit]:
            session_dir = path.parent
            rows = _rows(session_dir)
            chosen = clients.read(session_dir)
            audio = session_dir / archive.AUDIO_DIR
            found.append({
                "name": session_dir.name,
                "client": chosen.get("name") if chosen else None,
                "lines": len(rows),
                "minutes_long": round(max((float(row.get("end_time", 0)) for row in rows), default=0) / 60),
                "speakers": sorted({str(row.get("speaker", "")) for row in rows if row.get("speaker")}),
                "tracks": sorted(file.stem for file in audio.glob("*.mp3")) if audio.is_dir() else [],
                "has_minutes": (session_dir / "minutes.md").exists(),
                "minutes_url": (finish.read_state(session_dir).get("minutes") or {}).get("url", ""),
                "rebuilt": (session_dir / "transcripts_final.jsonl").exists(),
                "remaining": [{"key": step.key, "label": step.label}
                              for step in finish.remaining(session_dir, voice_library=self._voice_path(),
                                      voice_enabled=self._voice_enabled())],
                "wav_mb": round(archive.wav_bytes(session_dir) / 1048576),
            })
        return found

    def detail(self, session: str) -> dict:
        """会議 1 本の中身（会議中の画面と同じものを、あとから見るため）。"""
        session_dir = self._session(session)
        state = _read_json(session_dir / "state.json") or {}
        rows = _rows(session_dir)
        chosen = clients.read(session_dir)
        marks = finish.read_state(session_dir)
        minutes = session_dir / "minutes.md"
        return {
            "session": session,
            "client": chosen.get("name") if chosen else None,
            "participants": _participants(rows),
            "screens": _screens(session_dir),
            "pages": screen_capture.pages(session_dir),
            "state": {"summary": state.get("summary", ""),
                      "decisions": [_item(value) for value in state.get("decisions") or []],
                      "todos": [_item(value) for value in state.get("todos") or []],
                      "questions": [_item(value) for value in state.get("questions") or []],
                      "topics": [_item(value) for value in state.get("topics") or []]},
            "sections": _sections(state),
            "verifications": _read_lines(session_dir / "verifications.jsonl"),
            "minutes_url": (marks.get("minutes") or {}).get("url", ""),
            "minutes_note": (marks.get("minutes") or {}).get("note", ""),
            "minutes_text": minutes.read_text(encoding="utf-8") if minutes.exists() else "",
            # 「Notion に登録」を出してよいか。つないでいない環境では**押しても動かない**ので出さない
            #   （2026-09-21: 配布版は `notion_tasks: false` なのにボタンだけ出ていた）
            "can_register_minutes": bool((self._settings().get("task_hub") or {}).get("notion_tasks")),
            "files": self._files(session_dir),
            "remaining": [{"key": step.key, "label": step.label}
                          for step in finish.remaining(session_dir, voice_library=self._voice_path(),
                                      voice_enabled=self._voice_enabled())],
            **self.transcript(session),
        }

    def register_minutes(self, session: str) -> dict:
        """議事録を タスク管理 の議事録 DB へ流し込む（押されてから走る・運用者 決定 2026-09-17）。

        中身は `src/task_hub_minutes.py`（議事録の連鎖 の Step 3・3b・4・5 と同じ形）。
        哲学カードと製品 Fact の抽出は Claude CLI（テキストのみ・サブスクの範囲）。
        Claude CLI が無い環境では、議事録とタスクだけ登録して連鎖は飛ばす（理由を画面に出す）。
        """
        session_dir = self._session(session)
        # 二重登録の防止（2026-09-17 に気づいた）。押したあとに task_hub-minutes スキルを回すと
        #   議事録ページが 2 つできる。既に登録済みなら断り、どちらか片方にしてもらう
        already = (finish.read_state(session_dir).get("minutes") or {}).get("url", "")
        if already:
            raise ValueError(f"この会議の議事録は登録済みです: {already}（作り直すなら、先にその欄を空にしてください）")
        chosen = clients.read(session_dir)
        name = (chosen or {}).get("name") or clients.SELF_CLIENT
        settings = self._settings()
        shared = (settings.get("task_hub") or {}).get("shared_dir", "~/.claude/skills/_shared")
        meeting_date = self._meeting_date(session)
        claude = ClaudeCliClient()      # サブスクの範囲（従量課金は増えない）
        meeting = settings.get("meeting") or {}
        result = task_hub_minutes.register(
            session_dir, client_name=name, meeting_date=meeting_date, shared_dir=shared,
            llm=claude if claude.available() else None, prompts=self.repo / "prompts",
            self_name=str(meeting.get("self_name", "")),
            self_client_name=str((settings.get("task_hub") or {}).get("client_name", "")),
            model="claude-opus-5",      # 台帳に残す実行モデル名
            personas=settings_edit.read_personas(self._settings_path()),
            people_db=str((settings.get("task_hub") or {}).get("people_master_db", "")))
        finish.mark(session_dir, "minutes", note=f"{name} の議事録 DB に登録", url=result.url)
        return {"ok": True, "url": result.url, "client": name, "date": meeting_date,
                "linked": result.linked, "already": result.already, "skipped": result.skipped,
                "chains": result.chains}

    @staticmethod
    def _meeting_date(session: str) -> str:
        """会議の日（YYYY-MM-DD）。会議の名前が正本（`2026-09-17_0918`）。"""
        head = session.split("_")[0]
        return head if re.fullmatch(r"\d{4}-\d{2}-\d{2}", head) else datetime.now(JST).strftime("%Y-%m-%d")

    def set_minutes_url(self, session: str, url: str) -> dict:
        """議事録が外（Notion 等）にあることを覚える。以後「残り」に出さず、画面にリンクを出す。"""
        url = url.strip()
        if not url.startswith("http"):
            raise ValueError("議事録の URL を入れてください（http で始まるもの）")
        self._session(session)
        finish.mark(self._session(session), "minutes", note="外（Notion 等）に登録済み", url=url)
        return {"ok": True, "url": url}

    def _files(self, session_dir: Path) -> list[dict]:
        """ダウンロードできるもの（録音・議事録・全文）。"""
        found = []
        audio = session_dir / archive.AUDIO_DIR
        for track, label in (("meeting", "録音（統合）"), ("self", "録音（自分）"), ("remote", "録音（相手）")):
            path = audio / f"{track}.mp3"
            if path.exists():
                found.append({"kind": f"audio:{track}", "label": label,
                              "mb": round(path.stat().st_size / 1048576, 1)})
        for name, kind, label in (("minutes.md", "minutes", "議事録（Markdown）"),
                                  ("minutes_input.md", "input", "議事録の材料"),
                                  ("transcripts_final.jsonl", "transcript", "全文（作り直し）"),
                                  ("transcripts.jsonl", "transcript_live", "全文（会議中）")):
            path = session_dir / name
            if path.exists():
                found.append({"kind": kind, "label": label, "mb": round(path.stat().st_size / 1048576, 2)})
        return found

    def file_path(self, session: str, kind: str) -> Path:
        """ダウンロードするファイル。会議のフォルダの外は返さない。"""
        session_dir = self._session(session)
        names = {"minutes": "minutes.md", "input": "minutes_input.md",
                 "transcript": "transcripts_final.jsonl", "transcript_live": "transcripts.jsonl"}
        if kind.startswith("audio:"):
            return self.audio_file(session, kind.split(":", 1)[1])
        if kind not in names:
            raise KeyError(f"そのファイルはありません: {kind}")
        path = session_dir / names[kind]
        if not path.exists():
            raise KeyError(f"そのファイルはありません: {kind}")
        return path

    def transcript(self, session: str) -> dict:
        """その会議の全文（作り直したものがあればそちら）。時刻つきなので、聞き直す場所が分かる。"""
        session_dir = self._session(session)
        rows = _rows(session_dir)
        minutes = session_dir / "minutes.md"
        return {
            "session": session,
            "rebuilt": (session_dir / "transcripts_final.jsonl").exists(),
            "rows": [{"start": float(row.get("start_time", 0)), "end": float(row.get("end_time", 0)),
                      "speaker": str(row.get("speaker", "")), "text": str(row.get("text", "")),
                      "aizuchi": bool(row.get("aizuchi"))} for row in rows],
            "tracks": sorted(file.stem for file in (session_dir / archive.AUDIO_DIR).glob("*.mp3"))
            if (session_dir / archive.AUDIO_DIR).is_dir() else [],
            "minutes": minutes.read_text(encoding="utf-8") if minutes.exists() else "",
            "minutes_stale": minutes.exists() and (session_dir / EDITS_FILE).exists()
            and (session_dir / EDITS_FILE).stat().st_mtime > minutes.stat().st_mtime,
        }

    def edit_line(self, session: str, start: float, text: str = "", speaker: str = "") -> dict:
        """聞き直して、その行の文字（または話者）を直す。直した記録を残し、元の文も残す。"""
        session_dir = self._session(session)
        path = session_dir / ("transcripts_final.jsonl" if (session_dir / "transcripts_final.jsonl").exists()
                              else "transcripts.jsonl")
        rows = _rows(session_dir)
        target = next((row for row in rows if abs(float(row.get("start_time", 0)) - float(start)) < 0.05), None)
        if target is None:
            raise KeyError(f"その行がありません: {start}")
        before = {"text": str(target.get("text", "")), "speaker": str(target.get("speaker", ""))}
        if text:
            target["text"] = text
        if speaker:
            target["speaker"] = speaker
        if not text and not speaker:
            raise ValueError("直す中身がありません")
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        with (session_dir / EDITS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": datetime.now(JST).isoformat(timespec="seconds"), "start": float(start),
                                     "before": before, "after": {"text": text or before["text"],
                                                                 "speaker": speaker or before["speaker"]}},
                                    ensure_ascii=False) + "\n")
        return {"ok": True, "before": before}

    def screen_file(self, session: str, name: str) -> Path:
        """控えた画面の 1 枚。会議のフォルダの中だけ。"""
        session_dir = self._session(session)
        path = (session_dir / screen_capture.SCREENS_DIR / Path(name).name).resolve()
        if path.parent != (session_dir / screen_capture.SCREENS_DIR).resolve() or not path.exists():
            raise KeyError(f"その画面はありません: {name}")
        return path

    def audio_file(self, session: str, track: str) -> Path:
        """聞き直す音（mp3）。wav は大きいので画面には出さない。"""
        session_dir = self._session(session)
        path = session_dir / archive.AUDIO_DIR / f"{track}.mp3"
        if track not in {"meeting", "self", "remote"} or not path.exists():
            raise KeyError(f"その音はありません: {track}")
        return path

    # ------------------------------------------------------------------ 声の台帳

    def _voice_path(self) -> Path:
        config = (self._settings().get("meeting") or {}).get("voice_library") or {}
        return self._path(config.get("path", "workspace/voices/library.json"))

    def _voice_enabled(self) -> bool:
        """声の台帳を使う設定か。off なら「声を覚える」工程は最初から終わり扱い。"""
        config = (self._settings().get("meeting") or {}).get("voice_library") or {}
        return bool(config.get("enabled"))

    def rename_speaker(self, session: str, old: str, new: str, starts: list | None = None) -> dict:
        """会議のあとに、話者の名前を付ける／直す（不明話者に名前を付ける窓口）。

        会議中は行の ✎ で付けられるが、終わったあとの窓口が無かった（運用者 指摘 2026-09-17
        「不明話者3 はどこにもない」）。声の台帳には**名前の付いた人しか載らない**ので、
        ここで名前を付けてから覚える。

        やること: その会議の全文の話者名を書き換え → `renames.jsonl` に残す →
        （台帳を使っているなら）その人の声を覚える。

        `starts` を渡すと**その行だけ**直す。1 つの「不明話者」に**複数人が混ざっている**
        ことがあるため（声紋のまとまりが割れずに 1 人として出る）。混ざったまま一括で付けると、
        言っていない人の発言としてクライアントの台帳に残る。
        声の台帳は混ざりを見つけると覚えない（`learn` が「声がそろわない」で弾く）。
        その理由は画面に返す。
        """
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise ValueError("いまの名前と、新しい名前の両方を入れてください")
        if "不明話者" in new:
            raise ValueError("「不明話者」という名前は付けられません")
        session_dir = self._session(session)
        wanted = {round(float(value), 3) for value in (starts or [])}
        changed = 0
        for name in ("transcripts.jsonl", "transcripts_final.jsonl"):
            path = session_dir / name
            if not path.exists():
                continue
            lines, hit = [], 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                at = round(float(row.get("start_time", -1)), 3)
                if str(row.get("speaker", "")) == old and (not wanted or at in wanted):
                    row["speaker"] = new
                    hit += 1
                lines.append(json.dumps(row, ensure_ascii=False))
            if hit:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            changed = max(changed, hit)
        if not changed:
            raise KeyError(f"この会議に「{old}」の発言がありません")
        with (session_dir / "renames.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": datetime.now(JST).isoformat(), "old": old, "new": new,
                                     "how": "会議アシスタント", "rows": changed,
                                     "whole_speaker": not wanted}, ensure_ascii=False) + "\n")
        voices = self._learn_voice(session_dir)
        self._on_change("voices")
        return {"ok": True, "old": old, "new": new, "lines": changed,
                "learned": voices["learned"], "skipped": voices["skipped"]}

    def _learn_voice(self, session_dir: Path) -> dict:
        """その会議の声を台帳へ覚える（名前を付け直した直後に呼ぶ）。

        戻り値 `{"learned": [名前], "skipped": {名前: 理由}}`。
        覚えなかった理由（「声がそろわない＝複数人が混ざっている可能性」）を捨てない。
        """
        empty = {"learned": [], "skipped": {}}
        library, config = self._voices()
        if not config.enabled:
            return empty
        recording = session_dir / "recording_remote.wav"
        rows = _rows(session_dir)
        if not rows or not recording.exists():
            return empty
        try:
            from src.audio.enrolled_diarizer import EnrolledDiarizer
            from src.audio.voice_library import read_wav_mono, voices_from_session

            audio, sample_rate = read_wav_mono(recording)
            self_name = str((self._settings().get("meeting") or {}).get("self_name", ""))
            voices = voices_from_session(rows, audio, sample_rate, self_name=self_name,
                                         config=config, embed=EnrolledDiarizer.embed)
            result = library.learn(session_dir.name, voices)
            library.save()
            return {"learned": sorted(result.learned), "skipped": dict(result.skipped)}
        except Exception:                                   # noqa: BLE001 - 覚えられなくても改名は残す
            logger.warning("声を覚えられませんでした（改名は済んでいます）", exc_info=True)
            return empty

    def _voices(self):
        from src.audio.voice_library import VoiceLibrary, VoiceLibraryConfig

        config = VoiceLibraryConfig.from_mapping((self._settings().get("meeting") or {}).get("voice_library"))
        return VoiceLibrary(self._path(config.path), config), config

    def voices(self) -> dict:
        library, config = self._voices()
        people = [{"name": name, "meetings": len(entries), "sessions": [e.get("session", "") for e in entries],
                   "last": entries[-1].get("session", "") if entries else ""}
                  for name, entries in sorted(library.people.items())]
        return {"enabled": config.enabled, "path": str(library.path), "people": people}

    # ------------------------------------------------------- 設定（機能の ON/OFF・よく触る値）

    def settings_fields(self) -> dict:
        """画面から直せる設定の一覧（ここに挙げたものだけ）＋この Mac でできること。"""
        path = self._settings_path()
        found = capability.look(self._settings())
        return {"path": str(path), "fields": settings_edit.read_fields(path),
                "capability": found.as_dict(), "why_not_offline": found.why_not_offline(),
                "presets": settings_edit.presets_for(found),
                "preset": settings_edit.current_preset(path),
                "restart": "会議中の設定は、次の会議から効きます"}

    def apply_preset(self, name: str) -> dict:
        """動かし方（プリセット）を当てる。この Mac でできないものは当てない。"""
        result = settings_edit.apply_preset(self._settings_path(), name,
                                            capability.look(self._settings()),
                                            trash_dir=self.trash / "settings")
        self._on_change("settings")
        return {"ok": True, **result}

    SELF_NAMES_FILE = "previous_self_names.json"
    """名前を変える前の「自分の名前」を覚えておく場所。

    2026-09-18 運用者 報告「自分の名前を 自分 → 自社自分 に変えたら、哲学カードの人物が
    2 つになった」。古い会議の記録には前の名前が残るので、候補を集めると両方が並ぶ。
    同じ人だと機械には分からないので、変えた側で覚えておく。
    """

    def set_setting(self, key: str, value) -> dict:
        """設定を 1 つ直す。直す前のファイルはゴミ箱へ退避してから書く。"""
        before = None
        if key == "meeting.self_name":
            before = str((self._settings().get("meeting") or {}).get("self_name", "")).strip()
        saved = settings_edit.set_value(self._settings_path(), key, value,
                                        trash_dir=self.trash / "settings")
        if before and before != str(saved).strip():
            self._remember_old_self_name(before)
        self._on_change("settings")
        return {"ok": True, "key": key, "value": saved}

    def _previous_self_names(self) -> set[str]:
        """これまでに名乗っていた名前。読めなければ空（画面は出す）。"""
        path = self.repo / "workspace" / self.SELF_NAMES_FILE
        try:
            told = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        return {str(one).strip() for one in told if str(one).strip()} if isinstance(told, list) else set()

    def _remember_old_self_name(self, name: str) -> None:
        """前の名前を覚える。消さずに足すだけ（何度変えても並ばないように）。"""
        path = self.repo / "workspace" / self.SELF_NAMES_FILE
        names = self._previous_self_names() | {name}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(names), ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------- Notion のペルソナ名（哲学カードの行き先）

    def personas(self) -> dict:
        """話者名 → Notion のペルソナ名の対応表と、いま分かっている話者の一覧。

        運用者 依頼 2026-09-17「会議アシスタントの GUI からも設定できるように」。
        名前が合っていないと、その人の哲学カードは**登録されない**（誤登録を防ぐため止めている）。
        """
        table = settings_edit.read_personas(self._settings_path())
        meeting = self._settings().get("meeting") or {}
        self_name = str(meeting.get("self_name", ""))
        known = {self_name} if self_name else set()
        try:
            library, _ = self._voices()
            known |= set(library.people)
        except Exception:                                   # noqa: BLE001 - 台帳が無くても画面は出す
            pass
        for path in sorted(self.sessions_dir.glob("*/transcripts.jsonl"),
                           key=lambda item: item.stat().st_mtime, reverse=True)[:5]:
            known |= {str(row.get("speaker", "")) for row in _rows(path.parent) if row.get("speaker")}
        known |= set(table)
        # 前に名乗っていた名前は出さない（同じ人が 2 つ並ぶため）。対応表に載っている名前は残す
        known -= {name for name in self._previous_self_names() if name not in table}
        people = [{"speaker": name, "persona": "、".join(table.get(name, [])),
                   "personas": table.get(name, []), "self": name == self_name}
                  for name in sorted(known) if name and "不明話者" not in name]
        return {"people": people, "self_name": self_name, "path": str(self._settings_path())}

    def set_persona(self, speaker: str, persona: str) -> dict:
        """1 人ぶんの対応を足す／直す／外す（空にすると外す）。"""
        table = settings_edit.set_persona(self._settings_path(), speaker, persona,
                                          trash_dir=self.trash / "settings")
        self._on_change("personas")
        names = table.get(speaker, [])
        return {"ok": True, "speaker": speaker, "persona": "、".join(names), "personas": names}

    def _settings_path(self) -> Path:
        """設定ファイルの場所。見本しか無い環境でも落とさない。"""
        for name in ("settings.yaml", "settings.example.yaml"):
            path = self.repo / "config" / name
            if path.exists():
                return path
        return self.repo / "config" / "settings.yaml"

    def voice_sample(self, name: str) -> dict:
        """その人の声を確かめるための代表的な発言（長めで聞き取りやすい行）。

        「これは参加者Dさんだ」と耳で確かめられるようにする（運用者 依頼 2026-09-16）。
        台帳が覚えた会議の中から、**相手側の録音があって長めの行**を選ぶ。
        """
        library, _ = self._voices()
        entries = library.people.get(name)
        if not entries:
            raise KeyError(f"台帳にいません: {name}")
        for entry in reversed(entries):                     # 新しい会議から探す
            session_dir = self.sessions_dir / str(entry.get("session", ""))
            if not session_dir.is_dir() or not (session_dir / archive.AUDIO_DIR / "remote.mp3").exists():
                continue
            rows = [row for row in _rows(session_dir)
                    if str(row.get("speaker", "")) == name
                    and 3.0 <= float(row.get("end_time", 0)) - float(row.get("start_time", 0)) <= 12.0]
            if not rows:
                continue
            row = max(rows, key=lambda row: len(str(row.get("text", ""))))   # 文字数の多い＝はっきり話している行
            return {"name": name, "session": session_dir.name, "track": "remote",
                    "start": float(row["start_time"]), "end": float(row["end_time"]),
                    "text": str(row.get("text", ""))}
        raise KeyError(f"聞ける録音がありません: {name}（録音を mp3 に畳んだ会議が要ります）")

    def rename_voice(self, old: str, new: str) -> dict:
        new = new.strip()
        if not new:
            raise ValueError("新しい名前が空です")
        library, _ = self._voices()
        library.rename(old, new)
        library.save()
        self._on_change("voices")
        return {"ok": True}

    def forget_voice(self, name: str) -> dict:
        library, _ = self._voices()
        moved = library.forget(name, self.trash / "voices")
        library.save()
        self._on_change("voices")
        return {"ok": True, "backup": str(moved)}

    def _path(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.repo / path


def build_library_router(library: Library) -> APIRouter:
    router = APIRouter()
    page = Path(__file__).parent / "static" / "library.html"

    def run(call, *args):
        try:
            return JSONResponse(call(*args))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0] if error.args else error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except task_hub_minutes.NotionMinutesError as error:
            # 登録できなかった理由を、そのまま画面に出す（無音で失敗しない）
            raise HTTPException(status_code=502, detail=str(error)) from error

    @router.get("/library", response_class=HTMLResponse)
    def library_page() -> HTMLResponse:
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @router.get("/api/library/dictionaries")
    def dictionaries() -> JSONResponse:
        return run(library.dictionaries)

    @router.post("/api/library/dictionary/add")
    def add(payload: dict) -> JSONResponse:
        return run(library.add, str(payload.get("dictionary", "current")), str(payload.get("wrong", "")),
                   str(payload.get("right", "")))

    @router.post("/api/library/dictionary/remove")
    def remove(payload: dict) -> JSONResponse:
        return run(library.remove, str(payload.get("dictionary", "")), str(payload.get("wrong", "")))

    @router.post("/api/library/guard/add")
    def add_guard(payload: dict) -> JSONResponse:
        return run(library.add_guard, str(payload.get("dictionary", "current")), str(payload.get("word", "")))

    @router.post("/api/library/guard/remove")
    def remove_guard(payload: dict) -> JSONResponse:
        return run(library.remove_guard, str(payload.get("dictionary", "")), str(payload.get("word", "")))

    @router.get("/api/library/sessions")
    def sessions() -> JSONResponse:
        return run(library.sessions)

    @router.get("/api/library/candidates")
    def candidates(session: str) -> JSONResponse:
        return run(library.candidates, session)

    @router.post("/api/library/candidates/decide")
    def decide(payload: dict) -> JSONResponse:
        return run(library.decide, str(payload.get("session", "")), list(payload.get("items") or []))

    @router.get("/api/library/meetings")
    def meetings() -> JSONResponse:
        return run(library.meetings)

    @router.get("/api/library/meeting")
    def meeting(session: str) -> JSONResponse:
        return run(library.detail, session)

    @router.post("/api/library/minutes/register")
    def register_minutes(payload: dict) -> JSONResponse:
        return run(library.register_minutes, str(payload.get("session", "")))

    @router.post("/api/library/minutes")
    def set_minutes(payload: dict) -> JSONResponse:
        return run(library.set_minutes_url, str(payload.get("session", "")), str(payload.get("url", "")))

    @router.get("/api/library/file")
    def download(session: str, kind: str) -> Response:
        try:
            path = library.file_path(session, kind)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0] if error.args else error)) from error
        return FileResponse(path, filename=f"{session}-{path.name}", media_type="application/octet-stream")

    @router.get("/api/library/transcript")
    def transcript(session: str) -> JSONResponse:
        return run(library.transcript, session)

    @router.post("/api/library/transcript/edit")
    def edit_line(payload: dict) -> JSONResponse:
        return run(library.edit_line, str(payload.get("session", "")), float(payload.get("start", 0)),
                   str(payload.get("text", "")), str(payload.get("speaker", "")))

    @router.get("/api/library/screen")
    def screen(session: str, name: str) -> Response:
        try:
            path = library.screen_file(session, name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0] if error.args else error)) from error
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/api/library/audio")
    def audio(session: str, track: str = "meeting", request: Request = None) -> Response:
        """mp3 を返す（途中から聞けるように Range に答える）。"""
        try:
            path = library.audio_file(session, track)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0] if error.args else error)) from error
        size = path.stat().st_size
        header = (request.headers.get("range") if request is not None else None) or ""
        match = re.match(r"bytes=(\d+)-(\d*)", header)
        if not match:
            return Response(path.read_bytes(), media_type="audio/mpeg",
                            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else min(start + 1024 * 1024 - 1, size - 1)
        end = min(end, size - 1)
        with path.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(end - start + 1)
        return Response(chunk, status_code=206, media_type="audio/mpeg",
                        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                 "Accept-Ranges": "bytes", "Content-Length": str(len(chunk))})

    @router.get("/api/library/meeting-status")
    def meeting_status() -> JSONResponse:
        return run(library.meeting_status)

    @router.post("/api/library/start-meeting")
    def start_meeting() -> JSONResponse:
        return run(library.start_meeting)

    @router.get("/api/library/clients")
    def client_options() -> JSONResponse:
        return run(library.clients)

    @router.get("/api/library/unfinished")
    def unfinished() -> JSONResponse:
        return run(library.unfinished)

    @router.post("/api/library/meetings/delete")
    def delete_meetings(payload: dict) -> JSONResponse:
        return run(library.delete_meetings, [str(name) for name in (payload.get("sessions") or [])])

    @router.post("/api/library/meeting-client")
    def set_meeting_client(payload: dict) -> JSONResponse:
        return run(library.set_client, str(payload.get("session", "")), str(payload.get("client", "")))

    @router.post("/api/library/finish")
    def finish_session(payload: dict) -> JSONResponse:
        return run(library.finish_session, str(payload.get("session", "")), str(payload.get("client", "")))

    @router.get("/api/library/finish")
    def finish_status(session: str) -> JSONResponse:
        return run(library.finish_status, session)

    @router.get("/api/library/voices")
    def voices() -> JSONResponse:
        return run(library.voices)

    @router.get("/api/library/settings")
    def settings_fields() -> JSONResponse:
        return run(library.settings_fields)

    @router.post("/api/library/settings/preset")
    def apply_preset(payload: dict) -> JSONResponse:
        return run(library.apply_preset, str(payload.get("preset", "")))

    @router.post("/api/library/settings")
    def set_setting(payload: dict) -> JSONResponse:
        return run(library.set_setting, str(payload.get("key", "")), payload.get("value"))

    @router.get("/api/library/personas")
    def personas() -> JSONResponse:
        return run(library.personas)

    @router.post("/api/library/personas")
    def set_persona(payload: dict) -> JSONResponse:
        return run(library.set_persona, str(payload.get("speaker", "")), str(payload.get("persona", "")))

    @router.get("/api/library/voices/sample")
    def voice_sample(name: str) -> JSONResponse:
        return run(library.voice_sample, name)

    @router.post("/api/library/speaker/rename")
    def rename_speaker(payload: dict) -> JSONResponse:
        return run(library.rename_speaker, str(payload.get("session", "")),
                   str(payload.get("old", "")), str(payload.get("new", "")),
                   list(payload.get("starts") or []))

    @router.post("/api/library/voices/rename")
    def rename_voice(payload: dict) -> JSONResponse:
        return run(library.rename_voice, str(payload.get("old", "")), str(payload.get("new", "")))

    @router.post("/api/library/voices/forget")
    def forget_voice(payload: dict) -> JSONResponse:
        return run(library.forget_voice, str(payload.get("name", "")))

    return router


def task_hub_pusher(repo: Path, settings: Callable[[], dict]) -> Callable[[list[tuple[str, str]], str], list[dict]] | None:
    """入れた置き換えを タスク管理 の辞書へ「候補」で送る口（つないでいなければ None）。"""
    task_hub = settings().get("task_hub") or {}
    if not task_hub.get("dictionary_sync"):
        return None

    def push(pairs: list[tuple[str, str]], session: str) -> list[dict]:
        from src.text import task_hub_dictionary

        try:
            return task_hub_dictionary.add_candidates(task_hub.get("shared_dir", "~/.claude/skills/_shared"),
                                                   task_hub.get("client_name", "自社"), pairs)
        except Exception as error:  # noqa: BLE001 — タスク管理 に送れなくても辞書には入っている
            return [{"status": "error", "detail": str(error)[:200]}]

    return push
