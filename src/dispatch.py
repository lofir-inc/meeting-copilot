"""会議中の TODO をブリーフ、タスク管理、Agent へ払い出す。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.llm.meeting_state import Item, MeetingState
from src.task_hub import TaskHubBridge
from src.agent_client import AgentClient
from src.stt.whisper_client import TranscriptSegment
from src.text.aizuchi import is_aizuchi

logger = logging.getLogger(__name__)


@dataclass
class DispatchConfig:
    """払い出しの設定。"""

    enabled: bool = True
    context_sec: float = 90.0
    agent_repo: str = ""
    agent_command: str = "claude '{brief} を読み、このタスクに着手してください。完了したら {brief} の末尾に結果を追記してください'"
    agent_title: str = "▶ {task}"
    on_finish: str = "agent"
    minutes_repo: str = ""


@dataclass
class DispatchResult:
    """TODO 一件の各払い出し経路の結果。"""

    todo_id: str
    brief_path: Path
    already: bool = False
    notion: dict | None = None
    slack_ok: bool | None = None
    agent: dict | None = None
    errors: dict[str, str] = field(default_factory=dict)


def _session_value(session: dict, key: str, default: str = "") -> str:
    """後方互換のため複数のセッションキーから文字列を得る。"""
    aliases = {"client_name": ("client_name", "client"), "meeting_date": ("meeting_date", "date"), "session_name": ("session_name", "name")}
    for name in aliases.get(key, (key,)):
        if session.get(name) is not None:
            return str(session[name])
    return default


def _time_label(value: float) -> str:
    """会議秒を m:ss 表記へ変換する。"""
    return f"{int(value // 60)}:{int(value % 60):02d}"


def build_brief(state: MeetingState, todo: Item, segments: list[TranscriptSegment], session: dict, cfg: DispatchConfig) -> str:
    """着手者が単独で読める TODO ブリーフを構築する。"""
    owner = todo.by or "未定"
    nearby = [segment for segment in sorted(segments, key=lambda item: item.start_time) if abs(segment.start_time - todo.created_at) <= cfg.context_sec]
    decisions = [item for item in state.decisions if item.status == "open"]
    decisions.extend(sorted((item for item in state.decisions if item.status == "superseded"), key=lambda item: item.updated_at, reverse=True)[:3])
    questions = [item for item in state.questions if item.status == "open"]
    lines = [
        "# タスク",
        "",
        todo.text,
        "",
        "## 担当・期日",
        "",
        f"- 担当: {owner}",
        f"- 期日: {todo.due or '未定'}",
        "",
        "## 会議",
        "",
        f"- client: {_session_value(session, 'client_name', '（未設定）')}",
        f"- date: {_session_value(session, 'meeting_date', '（未設定）')}",
        f"- session: {_session_value(session, 'session_name', '（未設定）')}",
        "",
        "## 根拠の発話",
        "",
    ]
    lines.extend([f"[{_time_label(segment.start_time)}] {segment.speaker}: {segment.text}" for segment in nearby] or ["- （該当する発話なし）"])
    lines.extend(["", "## 関連する決定", ""])
    lines.extend([f"- [{item.id}] {item.text}（{item.status}）" for item in decisions] or ["- （なし）"])
    lines.extend(["", "## 未解決", ""])
    lines.extend([f"- [{item.id}] {item.text}" for item in questions] or ["- （なし）"])
    lines.extend(["", "## Notion", "", "- 登録後に URL を追記します。", ""])
    return "\n".join(lines)


def _was_dispatched(path: Path, todo_id: str) -> bool:
    """dispatch.jsonl から同一 TODO の払い出し済み状態を調べる。"""
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("todo_id") == todo_id:
                return True
        except json.JSONDecodeError:
            logger.warning("Invalid dispatch record ignored: %s", path)
    return False


def _append_record(path: Path, result: DispatchResult) -> None:
    """払い出し結果を JSONL へ追記する。"""
    value = asdict(result)
    value["brief_path"] = str(result.brief_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def dispatch(session_dir: Path, state: MeetingState, todo: Item, segments: list[TranscriptSegment], session: dict, cfg: DispatchConfig, bridge: TaskHubBridge | None, agent: AgentClient | None, owner: str) -> DispatchResult:
    """ブリーフ、Notion、Slack、Agent を順番に独立して実行する。"""
    dispatch_path = session_dir / "dispatch.jsonl"
    brief_path = session_dir / "dispatch" / f"{todo.id}.md"
    if _was_dispatched(dispatch_path, todo.id):
        return DispatchResult(todo_id=todo.id, brief_path=brief_path, already=True)
    result = DispatchResult(todo_id=todo.id, brief_path=brief_path)
    if not cfg.enabled:
        result.errors["dispatch"] = "dispatch is disabled"
        _append_record(dispatch_path, result)
        return result
    try:
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(build_brief(state, todo, segments, session, cfg), encoding="utf-8")
    except (OSError, ValueError) as exc:
        result.errors["brief"] = str(exc)
    if bridge is not None:
        try:
            result.notion = bridge.register_task(todo, brief_path.read_text(encoding="utf-8"), _session_value(session, "meeting_date"), owner)
            if result.notion.get("url"):
                with brief_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"- {result.notion['url']}\n")
        except Exception as exc:  # 外部連携の失敗は次経路を止めない。
            logger.warning("Notion dispatch failed for %s: %s", todo.id, exc)
            result.errors["notion"] = str(exc)
        try:
            notion_url = (result.notion or {}).get("url", "")
            message = f"▶ 着手: {todo.text}（担当 {owner}・期日 {todo.due or '未定'}）\n{notion_url}\n会議: {_session_value(session, 'session_name')}"
            result.slack_ok = bridge.notify(message)
        except Exception as exc:  # 外部連携の失敗は次経路を止めない。
            logger.warning("Slack dispatch failed for %s: %s", todo.id, exc)
            result.errors["slack"] = str(exc)
    if agent is not None and cfg.agent_repo:
        try:
            command = cfg.agent_command.format(brief=str(brief_path), task=todo.text)
            result.agent = agent.create_terminal(cfg.agent_repo, cfg.agent_title.format(task=todo.text), command)
        except Exception as exc:  # 外部連携の失敗は記録して終了する。
            logger.warning("Agent dispatch failed for %s: %s", todo.id, exc)
            result.errors["agent"] = str(exc)
    _append_record(dispatch_path, result)
    return result


def write_minutes_handoff(session_dir: Path, state: MeetingState, segments: list[TranscriptSegment], session: dict, dispatched: list[DispatchResult], corrections: list[dict] | None = None, mark_aizuchi: bool = True) -> tuple[Path, Path]:
    """タスク管理 議事録処理用の全文入力と引き渡しメタデータを書き出す。

    相づちだけの行には `(相づち)` を付ける（`speaker: text` の形は崩さない）。落としはしない
    — 議事録を作る側が読み飛ばせるようにするための印（運用者 決定 2026-09-13）。
    """
    corrections = corrections or []
    transcript_lines = []
    for segment in segments:
        speaker = segment.speaker
        for correction in corrections:
            if abs(float(correction.get("start_time", -999999)) - segment.start_time) <= 0.05:
                speaker = str(correction.get("new", speaker))
        mark = "(相づち) " if mark_aizuchi and is_aizuchi(segment.text) else ""
        transcript_lines.append(f"{speaker}: {mark}{segment.text}")
    todo_texts = {item.id: item.text for item in state.todos}
    registered = [result for result in dispatched if result.notion and result.notion.get("page_id")]
    minutes_input = session_dir / "minutes_input.md"
    lines = [
        f"# 議事録入力: {_session_value(session, 'session_name', session_dir.name)}",
        "",
        f"- client: {_session_value(session, 'client_name', '（未設定）')}",
        f"- date: {_session_value(session, 'meeting_date', '（未設定）')}",
        f"- session: {_session_value(session, 'session_name', session_dir.name)}",
        "",
        *_terms_section(session_dir),
        *_screens_section(session_dir),
        "## 話者つき全文",
        "",
        *transcript_lines,
        "",
        "## 会議中に着手済みタスク",
        "",
        "| Task Name | Notion URL | brief |",
        "| --- | --- | --- |",
    ]
    lines.extend([f"| {todo_texts.get(result.todo_id, result.todo_id)} | {result.notion.get('url', '')} | {result.brief_path} |" for result in registered] or ["| （なし） |  |  |"])
    lines.extend(["", "## 最終状態", "", state.to_markdown("最終状態")])
    minutes_input.write_text("\n".join(lines) + "\n", encoding="utf-8")
    handoff = session_dir / "minutes_handoff.json"
    payload = {
        "client_name": _session_value(session, "client_name"),
        "meeting_date": _session_value(session, "meeting_date"),
        "file_name": minutes_input.name,
        "transcript_path": str(minutes_input),
        "registered_tasks": [{"page_id": result.notion["page_id"], "url": result.notion.get("url", ""), "text": todo_texts.get(result.todo_id, result.todo_id)} for result in registered],
        "state_path": str(session_dir / "state.json"),
    }
    handoff.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Minutes handoff written: %s", handoff)
    return minutes_input, handoff


def finish_command(handoff: Path, cfg: DispatchConfig) -> list[str]:
    """タスク管理 議事録処理を起動する Agent CLI の引数配列を返す。"""
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    repo = cfg.minutes_repo or str(Path(__file__).resolve().parents[1])
    date = str(payload.get("meeting_date", ""))
    command = f"claude '/task_hub-minutes {handoff} を読んで議事録処理を実行。registered_tasks は新規作成せず Minutes リレーションで紐付けること'"
    return ["agent", "terminal", "create", "--worktree", f"path:{repo}", "--title", f"議事録 {date}", "--focus", "--command", command]


def _terms_section(session_dir: Path) -> list[str]:
    """事前資料の固有名詞（正しい表記の手がかり）。資料が無い会議では節ごと出さない。"""
    from src.text.terms import load_terms

    terms = load_terms(session_dir)
    if not terms:
        return []
    return ["## 事前資料の固有名詞（正しい表記の手がかり）", "", "、".join(terms), ""]


def _screens_section(session_dir: Path) -> list[str]:
    """画面共有で見せられたページ（相手は URL を読み上げない。あとで探せるように残す）。"""
    from src.screen.capture import pages

    found = pages(session_dir)
    if not found:
        return []
    lines = ["## 画面共有で見せられたページ", ""]
    for entry in found:
        clock = f"{int(entry['first_at']) // 60:02d}:{int(entry['first_at']) % 60:02d}"
        label = entry.get("title") or "（題名は読めず）"
        url = entry.get("url") or ""
        lines.append(f"- {clock} {label}{f' — {url}' if url else ''}")
    return lines + [""]
