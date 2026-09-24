"""議事録を タスク管理 の議事録 DB へ流し込む（議事録の連鎖 の Step 3・3b と同じ形）。"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from src import task_hub_minutes

MINUTES = """# 議事録: 見積の詰め

## 概要
- 配送の見え方を直す話が中心だった

## 決定事項
- 追跡番号の欄をトップへ上げる（田中）
- 公開は 7 月 10 日を目標にする

## TODO
| 誰が | 何を | いつまで |
|---|---|---|
| 田中 | 案を送る | 月曜 |

## 論点ごとの要点
### サイト改修の方針
- 問い合わせの半分が配送状況
- 追跡の導線を優先する

### 公開時期
- 検証を含めて 3 週間

## 未解決・次回確認
- 繁忙期の開始時期

## 出てきた数字・固有名詞
- 一日 30 件の問い合わせ
"""


class FakeNotion:
    """Notion 側の道具（`task_hub_notion`）の代わり。"""

    def __init__(self, pages: dict | None = None, writable: bool = True) -> None:
        self.pages = pages or {}
        self.writable = writable
        self.updates: list[tuple[str, str, dict]] = []
        self.ledger: list[dict] = []

    def get_page(self, page_id: str) -> dict:
        return self.pages.get(page_id, {"properties": {}})

    def relation_val(self, page_ids: list[str]) -> dict:
        return {"relation": [{"id": value} for value in page_ids]}

    def update_page_safe(self, page_id: str, db_id: str, props: dict):
        self.updates.append((page_id, db_id, props))
        return {"ok": True} if self.writable else None

    def log_local_run(self, skill: str, **kwargs) -> dict:
        self.ledger.append({"skill": skill, **kwargs})
        return {"ok": True}


class FakeMinutes:
    """`task_hub_minutes_lib` の代わり（作られたページと追記を覚える）。"""

    def __init__(self) -> None:
        self.created: dict = {}
        self.transcripts: list[tuple[str, str]] = []

    def build_minutes_markdown(self, draft: dict, client_name: str, meeting_date: str) -> str:
        body = f"# 議事録: {client_name} 打合せ ({meeting_date})\n\n"
        if draft.get("key_decisions"):
            body += "## ✅ 主要な意思決定\n" + "".join(f"- {d}\n" for d in draft["key_decisions"]) + "\n"
        for topic in draft.get("discussion_summary") or []:
            body += f"### {topic['topic']}\n- **論点**: {topic['discussion_points']}\n\n"
        if draft.get("open_items"):
            body += "## ❓ 未解決事項\n" + "".join(f"- {d}\n" for d in draft["open_items"]) + "\n"
        return body

    def create_minutes_page(self, minutes_db_id: str, client_name: str, meeting_date: str, markdown: str) -> dict:
        self.created = {"db": minutes_db_id, "client": client_name, "date": meeting_date, "markdown": markdown}
        return {"id": "page-1", "url": "https://notion.so/page-1"}

    def append_transcript(self, page_id: str, transcript: str) -> dict:
        self.transcripts.append((page_id, transcript))
        return {"ok": True}


def fake_shared(monkeypatch, *, minutes: FakeMinutes, notion: FakeNotion, client: dict) -> None:
    """`_shared` の読み込みを、作り物に差し替える。"""
    context = types.SimpleNamespace(resolve_client=lambda name=None: client)
    monkeypatch.setattr(task_hub_minutes, "_shared", lambda shared_dir: (context, minutes, notion))


def session_with(tmp_path: Path, *, minutes_text: str = MINUTES, rows: list[dict] | None = None,
                 handoff: list[dict] | None = None) -> Path:
    session = tmp_path / "2026-09-17_0918"
    session.mkdir()
    (session / "minutes.md").write_text(minutes_text, encoding="utf-8")
    for row in rows or []:
        with (session / "transcripts_final.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if handoff is not None:
        (session / "minutes_handoff.json").write_text(
            json.dumps({"registered_tasks": handoff}, ensure_ascii=False), encoding="utf-8")
    return session


class TestDraft:
    def test_議事録の見出しをWFの形に読み替える(self):
        draft = task_hub_minutes.draft_from_markdown(MINUTES)

        assert draft["key_decisions"] == ["追跡番号の欄をトップへ上げる（田中）", "公開は 7 月 10 日を目標にする"]
        assert [topic["topic"] for topic in draft["discussion_summary"]] == ["サイト改修の方針", "公開時期"]
        assert draft["open_items"] == ["繁忙期の開始時期"]
        assert "概要" in draft["_extras"] and "出てきた数字・固有名詞" in draft["_extras"]

    def test_形が違う議事録はそのまま流す(self):
        """手で書いた議事録や、昔の形でも登録できないより良い。"""
        text = "# メモ\n\nいろいろ話した。\n"

        assert task_hub_minutes.draft_from_markdown(text)["key_decisions"] == []
        assert task_hub_minutes.build_markdown(text, "取引先A", "2026-09-17", FakeMinutes()) == text

    def test_WFの形に組み直し余った節も落とさない(self):
        markdown = task_hub_minutes.build_markdown(MINUTES, "取引先A", "2026-09-17", FakeMinutes())

        assert "## ✅ 主要な意思決定" in markdown and "### サイト改修の方針" in markdown
        assert "## 概要" in markdown and "一日 30 件の問い合わせ" in markdown


class TestRegister:
    def test_議事録ページを作り全文を足しURLを返す(self, tmp_path, monkeypatch):
        session = session_with(tmp_path, rows=[{"speaker": "田中", "text": "よろしくお願いします"},
                                               {"speaker": "山田", "text": "はい"}])
        minutes, notion = FakeMinutes(), FakeNotion()
        fake_shared(monkeypatch, minutes=minutes, notion=notion,
                    client={"client_name": "取引先A", "minutes_db_id": "db-minutes",
                            "task_db_id": "db-tasks", "client_page_id": "client-1"})

        result = task_hub_minutes.register(session, client_name="取引先A", meeting_date="2026-09-17")

        assert result.url == "https://notion.so/page-1" and result.page_id == "page-1"
        assert minutes.created["db"] == "db-minutes" and minutes.created["date"] == "2026-09-17"
        assert minutes.transcripts == [("page-1", "田中: よろしくお願いします\n山田: はい")]
        assert notion.ledger and notion.ledger[0]["target_id"] == "page-1"

    def test_会議中に登録したタスクは作り直さず紐付ける(self, tmp_path, monkeypatch):
        session = session_with(tmp_path, handoff=[{"page_id": "task-1", "text": "案を送る"},
                                                  {"page_id": "task-2", "text": "原稿を用意する"}])
        notion = FakeNotion(pages={"task-2": {"properties": {"Minutes": {"relation": [{"id": "page-1"}]}}}})
        fake_shared(monkeypatch, minutes=FakeMinutes(), notion=notion,
                    client={"client_name": "取引先A", "minutes_db_id": "db-minutes", "task_db_id": "db-tasks"})

        result = task_hub_minutes.register(session, client_name="取引先A", meeting_date="2026-09-17")

        assert result.linked == 1 and result.already == 1 and result.skipped == []
        assert notion.updates[0][0] == "task-1" and notion.updates[0][1] == "db-tasks"

    def test_Minutes列が無ければ書けなかったと残す(self, tmp_path, monkeypatch):
        """無音で減らさない（update_page_safe は書けないと None を返すだけ）。"""
        session = session_with(tmp_path, handoff=[{"page_id": "task-1", "text": "案を送る"}])
        fake_shared(monkeypatch, minutes=FakeMinutes(), notion=FakeNotion(writable=False),
                    client={"client_name": "取引先A", "minutes_db_id": "db-minutes", "task_db_id": "db-tasks"})

        result = task_hub_minutes.register(session, client_name="取引先A", meeting_date="2026-09-17")

        assert result.skipped == [{"reason": "Minutes 列が無い", "page_id": "task-1"}]

    def test_議事録が無ければ断る(self, tmp_path, monkeypatch):
        session = tmp_path / "2026-09-17_0918"
        session.mkdir()
        fake_shared(monkeypatch, minutes=FakeMinutes(), notion=FakeNotion(), client={})

        with pytest.raises(task_hub_minutes.NotionMinutesError, match="minutes.md"):
            task_hub_minutes.register(session, client_name="取引先A", meeting_date="2026-09-17")

    def test_議事録DBが設定されていなければ断る(self, tmp_path, monkeypatch):
        session = session_with(tmp_path)
        fake_shared(monkeypatch, minutes=FakeMinutes(), notion=FakeNotion(),
                    client={"client_name": "取引先A", "minutes_db_id": ""})

        with pytest.raises(task_hub_minutes.NotionMinutesError, match="議事録 DB"):
            task_hub_minutes.register(session, client_name="取引先A", meeting_date="2026-09-17")

    def test_共有モジュールが無ければ言う(self, tmp_path):
        session = session_with(tmp_path)

        with pytest.raises(task_hub_minutes.NotionMinutesError, match="共有モジュール"):
            task_hub_minutes.register(session, client_name="取引先A", meeting_date="2026-09-17",
                                    shared_dir=tmp_path / "どこにもない")


def test_全文は作り直したほうを使う(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    (session / "transcripts.jsonl").write_text(
        json.dumps({"speaker": "田中", "text": "会議中の版"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (session / "transcripts_final.jsonl").write_text(
        json.dumps({"speaker": "田中", "text": "作り直した版"}, ensure_ascii=False) + "\n", encoding="utf-8")

    assert task_hub_minutes.transcript_text(session) == "田中: 作り直した版"


def test_共有モジュールの場所をsys_pathに足す(tmp_path):
    """タスク管理 の共有モジュールは `~/.claude/skills/_shared` にある（パッケージではない）。

    本物が読める環境（この Mac）では本物が返る。ここで見るのは「場所を通したか」だけ。
    """
    for name in ("task_hub_minutes_lib", "task_hub_context", "task_hub_notion"):
        (tmp_path / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")

    modules = task_hub_minutes._shared(tmp_path)

    assert len(modules) == 3 and str(tmp_path) in sys.path
    sys.path.remove(str(tmp_path))
