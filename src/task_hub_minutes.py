"""議事録（minutes.md）を タスク管理 の議事録 DB へ流し込む。議事録の連鎖 の Step 3／3b と同じ形で。

会議のあとに `minutes.md` はできるが、Notion への登録は手作業だった（URL を画面に貼る）。
ここは**押してから**走る（運用者 決定 2026-09-17「押してから。安定したら自動化したい」）。

やること（`~/.claude/skills/task_hub-minutes/SKILL.md` の Step 3・3b と同じ）:

1. クライアントを解決して議事録 DB を決める（会議ごとに選んだ相手。`client.json`）
2. `minutes.md` を WF と同じ節（✅ 主要な意思決定 ／ 💬 議論サマリー ／ ❓ 未解決事項）に組み直す
3. 議事録ページを作る（`Meeting_Name` / `Date` / `source_type=議事録`）
4. 全文を「🎙️ 声文字起こし」のトグルで足す
5. 会議中に登録済みのタスクを、**作り直さずに** `Minutes` リレーションで紐付ける
6. 使った回数の台帳に 1 行残す（失敗しても止めない）

7. 哲学カードと製品 Fact の連鎖（WF の Step 4・5）を走らせる（`src/task_hub_chains.py`）

このモジュールは**外部 API へ音声を送らない**。連鎖の抽出には Claude CLI（テキストのみ・
  サブスクの範囲）を使う＝会議の最終パスや裏取りと同じ経路で、関門の対象外。
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src import task_hub_chains

logger = logging.getLogger(__name__)

HEADING_DECISIONS = "決定事項"
HEADING_TOPICS = "論点ごとの要点"
HEADING_OPEN = "未解決・次回確認"
EXTRA_HEADINGS = ("概要", "出てきた数字・固有名詞")
"""WF の形に無い節。捨てずに末尾へ残す（せっかく書いたものを落とさない）。"""


class NotionMinutesError(RuntimeError):
    """議事録を登録できなかった（画面にそのまま出す）。"""


@dataclass
class Registration:
    """登録の結果。"""

    url: str
    page_id: str
    linked: int = 0
    already: int = 0
    skipped: list[dict] = field(default_factory=list)
    """書けなかったものは黙って減らさない（無音の失敗を作らない）。"""

    chains: dict = field(default_factory=dict)
    """哲学カードと製品 Fact の結果（`src/task_hub_chains.ChainResult`）。"""


def _shared(shared_dir: str | Path):
    """タスク管理 の共有モジュールを読む。無ければここで止める（画面に理由を出す）。"""
    path = Path(shared_dir).expanduser()
    if not (path / "task_hub_minutes_lib.py").is_file():
        raise NotionMinutesError(f"タスク管理 の共有モジュールがありません: {path}")
    if str(path) not in sys.path:
        sys.path.append(str(path))
    try:
        return (importlib.import_module("task_hub_context"),
                importlib.import_module("task_hub_minutes_lib"),
                importlib.import_module("task_hub_notion"))
    except ImportError as exc:                      # pragma: no cover - 環境依存
        raise NotionMinutesError(f"タスク管理 の共有モジュールを読めません: {exc}") from exc


def _bullets(block: str) -> list[str]:
    """`- ` の行だけを取り出す。"""
    return [line.lstrip("-*・ ").strip() for line in block.splitlines()
            if line.strip().startswith(("- ", "* ", "・"))]


def draft_from_markdown(text: str) -> dict:
    """`minutes.md` を WF の `minutes_draft` の形にする。

    見出しは `prompts/minutes.md` が決めているので、そこだけを見る。
    形が違う議事録（古いもの・手で書いたもの）でも落ちない: 拾えなければ空で返し、
    呼び出し側が本文をそのまま流す。
    """
    sections: dict[str, str] = {}
    current = ""
    for line in (text or "").splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip()
            sections[current] = ""
            continue
        if current:
            sections[current] += line + "\n"

    topics = []
    for raw in re.split(r"^###\s+", sections.get(HEADING_TOPICS, ""), flags=re.MULTILINE)[1:]:
        title, _, body = raw.partition("\n")
        points = _bullets(body)
        if title.strip():
            topics.append({"topic": title.strip(), "discussion_points": "／".join(points)})

    draft = {
        "key_decisions": _bullets(sections.get(HEADING_DECISIONS, "")),
        "discussion_summary": topics,
        "open_items": _bullets(sections.get(HEADING_OPEN, "")),
    }
    extras = {name: sections[name].strip() for name in EXTRA_HEADINGS if sections.get(name, "").strip()}
    if extras:
        draft["_extras"] = extras
    return draft


def build_markdown(minutes_text: str, client_name: str, meeting_date: str, task_hub_minutes_lib) -> str:
    """Notion へ流す本文。WF と同じ組み方（`build_minutes_markdown`）を通す。"""
    draft = draft_from_markdown(minutes_text)
    if not (draft.get("key_decisions") or draft.get("discussion_summary") or draft.get("open_items")):
        # 形を読み取れないときは、作った議事録をそのまま流す（登録できないより良い）
        return minutes_text
    markdown = task_hub_minutes_lib.build_minutes_markdown(draft, client_name, meeting_date)
    for name, body in (draft.get("_extras") or {}).items():
        markdown += f"## {name}\n{body}\n\n"
    return markdown


def transcript_text(session_dir: Path) -> str:
    """「話者: 発言」の形の全文。作り直した全文があればそちらを使う。"""
    for name in ("transcripts_final.jsonl", "transcripts.jsonl"):
        path = Path(session_dir) / name
        if not path.exists():
            continue
        lines = []
        for row in path.read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            try:
                value = json.loads(row)
            except json.JSONDecodeError:
                continue
            speaker, text = str(value.get("speaker", "")).strip(), str(value.get("text", "")).strip()
            if text:
                lines.append(f"{speaker}: {text}" if speaker else text)
        if lines:
            return "\n".join(lines)
    return ""


def registered_tasks(session_dir: Path) -> list[dict]:
    """会議中に Tasks DB へ登録済みのタスク（`minutes_handoff.json`）。"""
    path = Path(session_dir) / "minutes_handoff.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [task for task in (payload.get("registered_tasks") or []) if isinstance(task, dict)]


def link_tasks(tasks: list[dict], minutes_page_id: str, task_db_id: str, notion) -> Registration:
    """会議中に登録したタスクを議事録へ紐付ける（作り直さない・Status も触らない）。"""
    result = Registration(url="", page_id=minutes_page_id)
    for task in tasks:
        page_id = task.get("page_id")
        if not page_id:
            result.skipped.append({"reason": "page_id が無い", "text": str(task.get("text", ""))[:60]})
            continue
        if not task_db_id:
            result.skipped.append({"reason": "task_db_id が解決できない", "page_id": page_id})
            continue
        current = notion.get_page(page_id)
        have = [row["id"] for row in
                ((current.get("properties", {}).get("Minutes", {}) or {}).get("relation", []) or [])]
        if minutes_page_id in have:
            result.already += 1
            continue
        # update_page ではなく update_page_safe（Minutes 列が無いテナントでも 400 にしない）
        written = notion.update_page_safe(page_id, task_db_id,
                                          {"Minutes": notion.relation_val(have + [minutes_page_id])})
        if written:
            result.linked += 1
        else:
            result.skipped.append({"reason": "Minutes 列が無い", "page_id": page_id})
    return result


def register(session_dir: Path, *, client_name: str, meeting_date: str,
             shared_dir: str | Path = "~/.claude/skills/_shared",
             llm=None, prompts: Path | None = None, self_name: str = "",
             self_client_name: str = "", model: str = "", personas: dict | None = None,
             people_db: str = "") -> Registration:
    """議事録ページを作って、全文とタスクを紐付け、哲学カードと製品 Fact まで走らせる。

    押されてから呼ばれる。`llm` を渡さなければ連鎖（Step 4・5）は走らない。
    """
    session_dir = Path(session_dir)
    minutes = session_dir / "minutes.md"
    if not minutes.exists():
        raise NotionMinutesError("議事録（minutes.md）がまだありません。先に「仕上げる」を実行してください")
    context, task_hub_minutes_lib, notion = _shared(shared_dir)

    try:
        client = context.resolve_client(client_name)
    except Exception as exc:                        # noqa: BLE001 - 相手側の例外はそのまま見せる
        raise NotionMinutesError(f"クライアントを解決できません（{client_name}）: {exc}") from exc
    minutes_db_id = str(client.get("minutes_db_id") or "")
    if not minutes_db_id:
        raise NotionMinutesError(f"「{client_name}」に議事録 DB が設定されていません（Client Master を確認）")

    name = str(client.get("client_name") or client_name)
    markdown = build_markdown(minutes.read_text(encoding="utf-8"), name, meeting_date, task_hub_minutes_lib)
    try:
        page = task_hub_minutes_lib.create_minutes_page(minutes_db_id, name, meeting_date, markdown)
    except Exception as exc:                        # noqa: BLE001
        raise NotionMinutesError(f"議事録ページを作れませんでした: {exc}") from exc
    page_id = str(page.get("id", ""))
    url = str(page.get("url", "")) or (f"https://notion.so/{page_id.replace('-', '')}" if page_id else "")
    logger.info("Minutes page created: %s", url)

    try:                                            # 全文は入らなくても議事録は残す（fail-open）
        task_hub_minutes_lib.append_transcript(page_id, transcript_text(session_dir))
    except Exception:                               # noqa: BLE001
        logger.warning("全文の追記に失敗しました（議事録ページは作成済み）", exc_info=True)

    result = link_tasks(registered_tasks(session_dir), page_id,
                        str(client.get("task_db_id") or ""), notion)
    result.url, result.page_id = url, page_id

    if llm is not None and prompts is not None:
        result.chains = _run_chains(session_dir, llm=llm, prompts=prompts, client=client,
                                    client_name=name, minutes_page_id=page_id, minutes_url=url,
                                    self_name=self_name, self_client_name=self_client_name,
                                    model=model, context=context, task_hub_minutes_lib=task_hub_minutes_lib,
                                    notion=notion, shared_dir=shared_dir, personas=personas,
                                    people_db=people_db)

    try:                                            # 使った回数の台帳（失敗しても止めない）
        notion.log_local_run("realtime-minutes", title=f"{name}（{meeting_date}）",
                             target_id=page_id, client_page_id=client.get("client_page_id"))
    except Exception:                               # noqa: BLE001
        logger.debug("台帳への記録は飛ばしました", exc_info=True)
    return result


def _run_chains(session_dir: Path, *, llm, prompts: Path, client: dict, client_name: str,
                minutes_page_id: str, minutes_url: str, self_name: str, self_client_name: str,
                model: str, context, task_hub_minutes_lib, notion, shared_dir,
                personas: dict | None = None, people_db: str = "") -> dict:
    """哲学カードと製品 Fact（WF の Step 4・5）。どちらも失敗しても議事録は残す。"""
    transcript = transcript_text(session_dir)
    if not transcript:
        return {"notes": ["全文が無いので、哲学カードと製品 Fact は取りませんでした"]}
    speakers = {line.split(":", 1)[0].strip() for line in transcript.splitlines() if ":" in line}

    self_client = None
    if self_client_name and self_client_name != client_name:
        try:
            self_client = context.resolve_client(self_client_name)
        except Exception:                           # noqa: BLE001
            logger.warning("自社（%s）を解決できませんでした", self_client_name, exc_info=True)

    try:
        philosophy_lib = importlib.import_module("task_hub_philosophy")
    except ImportError:                             # pragma: no cover - 環境依存
        philosophy_lib = None

    merged = task_hub_chains.ChainResult()
    if philosophy_lib is None:
        merged.notes.append("哲学カードの登録モジュールが無いので飛ばしました")
    else:
        try:
            cards = task_hub_chains.philosophy(
                session_dir, llm=llm, prompts=prompts, transcript=transcript, speakers=speakers,
                minutes_page_id=minutes_page_id, minutes_url=minutes_url, client=client,
                self_client=self_client, self_name=self_name, context=context,
                philosophy_lib=philosophy_lib, task_hub_minutes_lib=task_hub_minutes_lib, notion=notion,
                model=model, personas=personas, people_db=people_db)
            merged.cards, merged.cards_skipped = cards.cards, cards.cards_skipped
            merged.notes += cards.notes
        except Exception as exc:                    # noqa: BLE001
            merged.notes.append(f"哲学カードの連鎖が途中で止まりました（{exc}）")

    try:
        facts = task_hub_chains.product_facts(
            session_dir, llm=llm, prompts=prompts, transcript=transcript,
            minutes_page_id=minutes_page_id, client=client, client_name=client_name,
            task_hub_minutes_lib=task_hub_minutes_lib)
        merged.facts, merged.facts_skipped = facts.facts, facts.facts_skipped
        merged.notes += facts.notes
    except Exception as exc:                        # noqa: BLE001
        merged.notes.append(f"製品 Fact の連鎖が途中で止まりました（{exc}）")
    return merged.as_dict()
