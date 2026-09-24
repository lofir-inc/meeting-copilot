"""会議のローリング状態と LLM 差分を扱う。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Status = Literal["open", "done", "dropped", "answered", "superseded", "closed", "active", "asked", "skipped"]
_CLOSED_STATUSES = {"done", "dropped", "answered", "superseded", "closed", "asked", "skipped"}


@dataclass
class Item:
    """決定、TODO、質問、論点、次に聞くことを表す状態項目。"""

    id: str
    text: str
    status: Status = "open"
    by: str = ""
    due: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    note: str = ""


@dataclass
class MeetingState:
    """コードが保持する会議の累積状態。"""

    version: int = 1
    updated_at: float = 0.0
    current_topic: str = ""
    summary: str = ""
    topics: list[Item] = field(default_factory=list)
    decisions: list[Item] = field(default_factory=list)
    todos: list[Item] = field(default_factory=list)
    questions: list[Item] = field(default_factory=list)
    asks: list[Item] = field(default_factory=list)
    next_asks: list[str] = field(default_factory=list)

    def finalize(self, at: float) -> list[str]:
        """会議終了時に、聞かれないまま残った問いを skipped に確定して変更行を返す。"""
        changes: list[str] = []
        for item in self.asks:
            if item.status == "open":
                item.status = "skipped"
                item.updated_at = at
                changes.append(f"~ [{item.id}] {item.text}（open → skipped）")
        if changes:
            self.updated_at = max(self.updated_at, at)
            self.next_asks = []
        return changes

    def to_dict(self) -> dict:
        """JSON 化できる辞書へ変換する。"""
        return asdict(self)

    def to_json(self) -> str:
        """状態を JSON 文字列へ変換する。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, source: str) -> "MeetingState":
        """JSON 文字列から状態を復元する。"""
        data = json.loads(source)
        return cls(
            version=int(data.get("version", 1)),
            updated_at=float(data.get("updated_at", 0.0)),
            current_topic=str(data.get("current_topic", "")),
            summary=str(data.get("summary", "")),
            topics=[Item(**item) for item in data.get("topics", [])],
            decisions=[Item(**item) for item in data.get("decisions", [])],
            todos=[Item(**item) for item in data.get("todos", [])],
            questions=[Item(**item) for item in data.get("questions", [])],
            asks=[Item(**item) for item in data.get("asks", [])],
            next_asks=[str(item) for item in data.get("next_asks", [])][:3],
        )

    def to_prompt_text(self, max_chars: int = 3000) -> str:
        """LLM に渡すための、人が読める圧縮状態を返す。"""
        recent_after = self.updated_at - 600.0

        def visible(items: list[Item]) -> list[Item]:
            return [item for item in items if item.status not in _CLOSED_STATUSES or item.updated_at >= recent_after]

        def line(item: Item, text_chars: int = 80, note_chars: int = 60) -> str:
            """プロンプト用の 1 行。本文と注記は切る（TODO 69 件で 13,551 字になった実測 2026-09-08）。"""
            parts = [f"- [{item.id}] {item.text[:text_chars]}（{item.status}）"]
            if item.by:
                parts.append(f"担当・発言者: {item.by}")
            if item.due:
                parts.append(f"期日: {item.due}")
            if item.note:
                parts.append(f"注記: {item.note[:note_chars]}")
            return " / ".join(parts)

        topics = sorted(self.topics, key=lambda item: item.created_at, reverse=True)[:3]
        # 08-26 実測: 状態・担当・注記まで出すと gemma4 がその行を丸ごと next_asks に写す（"[A8] [A4] [A3] 本文（asked）"）。
        #   open だけを本文のみで出す。
        asks = sorted((item for item in self.asks if item.status == "open"), key=lambda item: item.created_at, reverse=True)[:8]
        # TODO は「担当が要る」ので全 open を載せていたが、08-26 で 69 件・13,551 字に膨らみ
        #   state の 75% を占めた。open は新しい 20 件だけ載せ、残りは件数で示す（closed は直近 10 分のまま）。
        summary = self.summary[:300]

        def compact_open(items: list[Item], limit: int, noun: str) -> list[str]:
            open_items = [item for item in items if item.status == "open"]
            closed_items = [item for item in visible(items) if item.status != "open"]
            recent_open = sorted(open_items, key=lambda item: item.created_at, reverse=True)[:limit]
            lines = [line(item) for item in closed_items + recent_open]
            remaining = len(open_items) - len(recent_open)
            if remaining:
                lines.append(f"- 他 {remaining} 件の{noun}（state.json 参照）")
            return lines

        decision_lines = compact_open(self.decisions, 8, "決定")
        question_lines = compact_open(self.questions, 5, "質問")
        todo_lines = compact_open(self.todos, 20, "TODO")

        def render() -> str:
            sections = [f"現在の論点: {self.current_topic or '（未設定）'}", f"要約: {summary or '（なし）'}"]
            topic_lines = [f"- [{item.id}] {item.text[:80]}（{int(item.created_at // 60)}:{int(item.created_at % 60):02d}〜）" for item in topics]
            sections.append("論点:\n" + ("\n".join(topic_lines) if topic_lines else "- （なし）"))
            sections.append("決定:\n" + ("\n".join(decision_lines) if decision_lines else "- （なし）"))
            sections.append("TODO:\n" + ("\n".join(todo_lines) if todo_lines else "- （なし）"))
            sections.append("未解決の質問:\n" + ("\n".join(question_lines) if question_lines else "- （なし）"))
            sections.append("次に聞くこと（未）:\n" + ("\n".join(f"- [{item.id}] {item.text[:80]}" for item in asks) if asks else "- （なし）"))
            return "\n\n".join(sections)

        text = render()
        if len(text) <= max_chars:
            return text
        summary = summary[:max(0, len(summary) - (len(text) - max_chars))]
        text = render()
        for collection in (topics, question_lines, todo_lines):
            while len(text) > max_chars and collection:
                # 末尾の「他 N 件」行は残し、その手前（＝一番古い open）から落とす
                last = collection[-1]
                keeps_count_line = isinstance(last, str) and last.startswith("- 他 ") and len(collection) > 1
                index = len(collection) - 2 if keeps_count_line else len(collection) - 1
                collection.pop(index)
                text = render()
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def to_markdown(self, title: str) -> str:
        """議事録用の Markdown を論点の流れを含む区画で返す。"""
        def section(name: str, items: list[Item], todo: bool = False) -> str:
            lines = [f"## {name}", ""]
            if not items:
                lines.append("- （なし）")
            for item in items:
                details = []
                if item.by:
                    details.append(f"担当・発言者: {item.by}")
                if todo and item.due:
                    details.append(f"期日: {item.due}")
                if item.note:
                    details.append(f"注記: {item.note}")
                suffix = f" — {' / '.join(details)}" if details else ""
                lines.append(f"- [{item.id}] {item.text}（{item.status}）{suffix}")
            return "\n".join(lines)

        topic_lines = [f"- [{item.id}] {int(item.created_at // 60)}:{int(item.created_at % 60):02d} {item.text}（{item.status}）" for item in sorted(self.topics, key=lambda item: item.created_at, reverse=True)]
        ask_marks = {"open": "○", "asked": "✓", "skipped": "—"}
        active_topic_id = self.topics[-1].id if self.topics else ""
        ask_lines = []
        for item in sorted(self.asks, key=lambda item: item.created_at, reverse=True):
            note = f"（{item.note}）" if item.note else ""
            mark = "△" if item.status == "open" and item.by and item.by != active_topic_id else ask_marks.get(item.status, "—")
            ask_lines.append(f"- {mark} [{item.id}] {item.text}{note}")

        return "\n\n".join([
            f"# {title}",
            "## 論点の流れ\n\n" + ("\n".join(topic_lines) if topic_lines else "- （なし）"),
            f"## 要約\n\n{self.summary or '（要約なし）'}",
            section("決定事項", self.decisions),
            section("TODO", self.todos, todo=True),
            section("未解決の質問", self.questions),
            "## 次に聞くべきこと\n\n" + ("\n".join(ask_lines) if ask_lines else "- （なし）"),
        ]) + "\n"


@dataclass
class StateDelta:
    """LLM が返す状態差分。"""

    summary: str
    current_topic: str
    new_decisions: list[dict]
    new_todos: list[dict]
    new_questions: list[dict]
    updates: list[dict]
    next_asks: list[str]
    topic_changed: bool = False


# 論点の切り替わりは LLM に判定させる（3e の実測: 文字列の類似では 1 時間に 40〜46 件立った。
#   「検索パフォーマンスと生成AIの連携」→「生成AIによる検索順位向上施策」のような言い換えは意味でしか見分けられない）。
#   コードは床だけ持つ: 直前の切り替えから MIN_TOPIC_SEC 未満なら切り替えない（往復の揺れを吸収）。
MIN_TOPIC_SEC = 90.0
# 問いは論点ごとに open を 3 件まで（08-26 実測: 1 窓 0〜2 件ずつ 228 窓で 134 件に積み上がった。1 回の生成は暴走していない。
#   論点 1 つは 2〜4 分なので、その間に司会者が消化できる問いは 3 件が上限）。
MAX_OPEN_ASKS_PER_TOPIC = 3

DELTA_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "current_topic", "topic_changed", "new_decisions", "new_todos", "new_questions", "updates", "next_asks"],
    "properties": {
        "summary": {"type": "string"}, "current_topic": {"type": "string"}, "topic_changed": {"type": "boolean"},
        "new_decisions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "by"], "properties": {"text": {"type": "string"}, "by": {"type": "string"}}}},
        "new_todos": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "owner", "due"], "properties": {"text": {"type": "string"}, "owner": {"type": "string"}, "due": {"type": "string"}}}},
        "new_questions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "asked_by"], "properties": {"text": {"type": "string"}, "asked_by": {"type": "string"}}}},
        "updates": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "status", "note"], "properties": {"id": {"type": "string"}, "status": {"type": "string", "enum": ["done", "dropped", "answered", "superseded", "closed", "asked"]}, "note": {"type": "string"}}}},
        "next_asks": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    },
}


def _normalize(text: str) -> str:
    return re.sub(r"\W", "", text).casefold()


def _similar(left: str, right: str) -> bool:
    left_normalized, right_normalized = _normalize(left), _normalize(right)
    if not left_normalized or not right_normalized:
        return left_normalized == right_normalized
    if left_normalized == right_normalized:
        return True
    left_bigrams = {left_normalized[index:index + 2] for index in range(len(left_normalized) - 1)}
    right_bigrams = {right_normalized[index:index + 2] for index in range(len(right_normalized) - 1)}
    return bool(left_bigrams | right_bigrams) and len(left_bigrams & right_bigrams) / len(left_bigrams | right_bigrams) >= 0.6


_TOPIC_PARTICLES = re.compile(r"について|に関して|に関する|の件|の話|とは|の|は|が|を|に|で|と|も|へ|や")


def _topic_core(text: str) -> str:
    """論点の芯（助詞や「について」を落とした正規化文字列）を返す。"""
    return _TOPIC_PARTICLES.sub("", _normalize(text))


def _same_topic(left: str, right: str) -> bool:
    """論点の言い換え（「予算の配分」と「予算配分について」）を同じ論点とみなす。"""
    # _similar（bigram Jaccard 0.6）だけだと 30 秒ごとの言い換えで論点が増え、問いが全部スルーに落ちる。
    #   包含（片方の正規化文字列がもう片方に入る）か、ゆるい Jaccard 0.4 で同一扱いにする。
    left_normalized, right_normalized = _topic_core(left), _topic_core(right)
    if not left_normalized or not right_normalized:
        return left_normalized == right_normalized
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return True
    left_bigrams = {left_normalized[index:index + 2] for index in range(len(left_normalized) - 1)}
    right_bigrams = {right_normalized[index:index + 2] for index in range(len(right_normalized) - 1)}
    return bool(left_bigrams | right_bigrams) and len(left_bigrams & right_bigrams) / len(left_bigrams | right_bigrams) >= 0.4


_ASK_TAG = re.compile(r"^(\s*\[[A-Z]\d+\]\s*)+")
_ASK_SUFFIX = re.compile(r"[（(]\s*(open|asked|skipped|answered|done|closed|superseded|dropped)\s*[）)]\s*$")


def _clean_ask(text: str) -> str:
    """LLM が写した id タグ（[A3] [Q1]）と状態の括弧を落とし、問いの本文だけにする。"""
    cleaned = _ASK_TAG.sub("", text.strip())
    while _ASK_SUFFIX.search(cleaned):
        cleaned = _ASK_SUFFIX.sub("", cleaned).strip()
    cleaned = cleaned.split(" / 担当・発言者:")[0].split(" / 注記:")[0].strip()
    return cleaned


def _next_id(items: list[Item], prefix: str) -> str:
    values = [int(match.group(1)) for item in items if (match := re.fullmatch(rf"{prefix}(\d+)", item.id))]
    return f"{prefix}{max(values, default=0) + 1}"


def apply_delta(prev: MeetingState, delta: StateDelta, at: float) -> tuple[MeetingState, list[str]]:
    """差分を適用し、追加または変更された項目の表示行を返す。"""
    state = MeetingState.from_json(prev.to_json())
    state.updated_at = at
    summary = delta.summary.strip()
    current_topic = delta.current_topic.strip()
    if summary:
        state.summary = summary[:300]
    changes: list[str] = []
    topic_text = current_topic or state.current_topic
    # active を探すと、LLM が updates で P を superseded にした瞬間に「最初の論点」扱いになり同文の P が二重に立つ（08-26 実測）。
    #   時系列の末尾を直前の論点とみなし、P は LLM の updates の対象から外す。
    previous_topic = state.topics[-1] if state.topics else None
    first_topic = bool(topic_text) and previous_topic is None
    # LLM が「切り替わった」と言い、かつ床（MIN_TOPIC_SEC）を越えていて、同じ論点の言い換えでもないときだけ切る
    changed = bool(current_topic) and delta.topic_changed and previous_topic is not None and at - previous_topic.created_at >= MIN_TOPIC_SEC and not _same_topic(current_topic, previous_topic.text)
    if first_topic or changed:
        if previous_topic is not None:
            previous_topic.status = "closed"
            previous_topic.updated_at = at
        # 論点が移っても問いは open のまま残す（08-26 実測: 論点の中央継続 152 秒＝5 窓で「聞かれる前に skipped」が 7 割）。
        #   「聞かずにスルーした」は会議終了時（finalize）に確定する。UI は closed な論点の下の open を △ で見せる。
        topic = Item(id=_next_id(state.topics, "P"), text=topic_text, status="active", created_at=at, updated_at=at)
        state.topics.append(topic)
        changes.append(f"+ [{topic.id}] {topic.text}（{topic.status}）")
    if current_topic:
        state.current_topic = current_topic
    collections = (state.decisions, state.todos, state.questions)
    all_items = [item for group in collections for item in group]

    def add(collection: list[Item], prefix: str, value: dict, by_key: str, todo: bool = False) -> None:
        text = str(value["text"]).strip()
        if not text or any(_similar(text, item.text) for item in all_items):
            return
        item = Item(id=_next_id(collection, prefix), text=text, status="open", by=str(value[by_key]).strip(), due=str(value.get("due", "")).strip(), created_at=at, updated_at=at)
        if todo:
            item.by = item.by or "未定"
            item.due = item.due or "未定"
        collection.append(item)
        all_items.append(item)
        changes.append(f"+ [{item.id}] {item.text}（{item.status}）")

    for value in delta.new_decisions:
        add(state.decisions, "D", value, "by")
    for value in delta.new_todos:
        add(state.todos, "T", value, "owner", todo=True)
    for value in delta.new_questions:
        add(state.questions, "Q", value, "asked_by")

    by_id = {item.id: item for item in [*all_items, *state.asks]}
    for update in delta.updates:
        item = by_id.get(str(update["id"]))
        if item is None:
            warning = f"⚠ 未知の id: {update['id']}"
            logger.warning(warning)
            changes.append(warning)
            continue
        old_status, old_note = item.status, item.note
        status = str(update["status"])
        note = str(update["note"]).strip()
        if item.id.startswith("A"):
            # 問い（A）を LLM が閉じられるのは「聞かれて答えが出た」だけ。dropped・superseded（「トピックが変更されたため一旦保留」）は
            # 無視して open のまま残す＝△。skipped は会議終了時（finalize）にしか付かない（aa 実測 2026-09-09: 4 分で 3 件が赤になっていた）。
            if status not in {"asked", "answered", "done", "closed"}:
                logger.info("問い %s への %s は無視（open のまま）: %s", item.id, status, note[:40])
                continue
            status = "asked"
            # 注記が問いの本文を丸ごと写していたら、本文を落として括弧の中だけ残す（同じ文が 2 行続いて見える）
            if note.startswith(item.text):
                note = note[len(item.text):].strip().strip("（）()").strip()
        item.status = status
        item.note = note
        item.updated_at = at
        if item.status != old_status or item.note != old_note:
            changes.append(f"~ [{item.id}] {item.text}（{old_status} → {item.status}）")
    current_topic_id = next((item.id for item in reversed(state.topics) if item.status == "active"), "")
    for text in delta.next_asks:
        ask_text = _clean_ask(text)
        if not ask_text:
            continue
        if sum(1 for item in state.asks if item.status == "open" and item.by == current_topic_id) >= MAX_OPEN_ASKS_PER_TOPIC:
            break
        # 既存の問い（状態を問わず全期間）と参加者の質問（Q）に似ていれば足さない。
        #   08-26 実測: 15 分窓だと古い問いを LLM が写して A5→A6→A7→A8 と同文が並んだ。skipped は終了時にしか付かないので open に戻す経路は不要。
        if any(_similar(ask_text, item.text) for item in [*state.asks, *state.questions]):
            continue
        ask = Item(id=_next_id(state.asks, "A"), text=ask_text, status="open", by=current_topic_id, created_at=at, updated_at=at)
        state.asks.append(ask)
        changes.append(f"+ [{ask.id}] {ask.text}（{ask.status}）")
    state.next_asks = [item.text for item in sorted((item for item in state.asks if item.status == "open"), key=lambda item: item.created_at, reverse=True)[:3]]
    return state, changes
