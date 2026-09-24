"""事前資料 — 会議ごとに空から始め、画面から足す。長いものは 1 回だけまとめる。

2026-09-11 の事故の裏返し: 事前資料は会議をまたぐ共有フォルダに置かれていて、4 月の別会議の
インタビュー計画が残ったまま定例の評価基準になっていた。いまは**会議ごとに空**から始め、
要るものだけをその場で足す（運用者 依頼 2026-09-14）。

長い資料は「足したときに 1 回だけ」まとめる。実物の提案書 PDF は 102,207 文字あり、
毎回の窓のプロンプトに入るのは先頭 2,000 文字だけなので、そのままでは**資料の 2% しか見ていない**。
全文を毎窓に載せると 30 秒ごとに課金され、手元のモデルでは文脈にも入らない。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.text.documents import (
    DOCUMENT_SUFFIXES,
    TEXT_SUFFIXES,
    UnreadableDocument,
    extract_text,
)

logger = logging.getLogger(__name__)

PREP_SUFFIXES = TEXT_SUFFIXES | DOCUMENT_SUFFIXES
"""画面から足せる種類。文字のファイルと、本文を取り出せる資料（PDF・Word・Excel・PowerPoint）。"""

PREP_MAX_CHARS = 500_000
"""取り出した本文の上限。要約へ渡るのは先頭だけ（`meeting.state.prior_tasks_chars`）。"""

PREP_MAX_BYTES = 40 * 1024 * 1024
"""受け取るファイルの上限。37 MB の提案書 PDF が実在したので、それが通る幅にしてある。"""

PREP_ORIGINALS = "originals"
"""元のファイルを残す場所（セッションの prep の下）。何を材料にしたかを後から辿るため。"""

PREP_DIGEST_MAX_INPUT = 200_000
"""まとめるときに読ませる文字数の上限（外のエンジン）。超えるぶんは頭から読んで打ち切る。"""

PREP_DIGEST_LOCAL_INPUT = 30_000
"""手元（Ollama）でまとめるときの上限。

`PREP_DIGEST_NUM_CTX` に収まる量で切る。日本語はおおよそ 1 文字 1 トークンなので、
これを超えて渡しても**黙って頭から捨てられる**（読んだつもりで読んでいない状態になる）。
"""

PREP_DIGEST_NUM_CTX = 32768
PREP_DIGEST_NUM_PREDICT = 2000
"""まとめの 1 回だけは、窓ごとの設定より大きく取る。

2026-09-14 実測: 13,028 文字の Word を窓の設定（8192 / 700）でまとめさせると、
**JSON が途中で切れて読めない**。32768 / 2000 なら通る（gemma4 で 26 秒）。
"""

PREP_DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}},
        "numbers": {"type": "array", "items": {"type": "string"}},
        "tasks": {"type": "array", "items": {"type": "string"}},
        "terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["purpose", "points", "numbers", "tasks", "terms"],
}

EMPTY = "（前回タスク・アジェンダ未設定）"


def safe_name(name: str) -> str:
    """画面から来た名前を、セッションの中だけに収まるファイル名にする。"""
    base = Path(str(name)).name.strip()
    base = base.replace("/", "_").replace("\\", "_").lstrip(".")
    if not base:
        raise ValueError("名前がありません")
    return base[:120]


def render_digest(data: dict, original_chars: int, limit: int) -> str:
    """まとめを、会議中ずっと載せる短いメモに整える。

    長さは**こちらで詰める**。1,200 文字以内と指示しても、実測（gemma4・2026-09-14）では
    2,800 文字返ってきた。溢れるときは「確認したい論点」の後ろから落とす
    — 前回タスク・数字・用語は照合に使うので残す。
    """
    def bullets(key: str) -> list[str]:
        return [f"- {str(value).strip()}" for value in (data.get(key) or []) if str(value).strip()]

    head = [f"### 事前資料のまとめ（自動・元は {original_chars:,} 文字）"]
    purpose = str(data.get("purpose", "")).strip()
    if purpose:
        head.append(f"目的: {purpose}")
    sections: dict[str, list[str]] = {}
    for key, label in (("tasks", "前回タスク・宿題"), ("points", "確認したい論点"),
                       ("numbers", "数字・期日"), ("terms", "用語")):
        items = bullets(key)
        if items:
            sections[key] = [f"\n**{label}**"] + items

    def rendered() -> str:
        lines = list(head)
        for key in ("tasks", "points", "numbers", "terms"):
            lines += sections.get(key, [])
        return "\n".join(lines)

    while len(rendered()) > limit and len(sections.get("points", [])) > 1:
        sections["points"] = sections["points"][:-1]        # 論点の後ろから落とす
    if len(sections.get("points", [])) <= 1:
        sections.pop("points", None)
    return rendered()[:limit]


class PrepMaterials:
    """この会議の事前資料（読み込み・追加・削除・まとめ）。

    オーケストレーターは「中身（`text` / `for_prompt`）」と「変わったこと（`on_change`）」だけを見る。
    """

    def __init__(
        self,
        session_dir: Path,
        prompts_dir: Path,
        *,
        limit: Callable[[], int],
        llm: Callable[[], object],
        engine: Callable[[], str],
        on_change: Callable[[], None] | None = None,
        publish: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.prompts_dir = Path(prompts_dir)
        self._limit = limit
        self._llm = llm
        self._engine = engine
        self._on_change = on_change
        self._publish = publish
        self.text: str = ""
        """読み込んだ全文（資料が無ければ `EMPTY`）。"""
        self.for_prompt: str = ""
        self.digest_terms: list[str] = []
        """長い資料をまとめたときに LLM が挙げた固有名詞（`prompts/prep_digest.md` の terms）。"""
        """毎窓のプロンプトへ載せるほう。長い資料は 1 回だけまとめたもの。"""

    @property
    def limit(self) -> int:
        """毎窓のプロンプトへ入れる文字数の上限。設定は会議中にも変わりうるので、**都度読む**。"""
        return self._limit()

    # ---------------------------------------------------------------- 置き場

    def directory(self) -> Path:
        path = self.session_dir / "prep"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def files(self) -> list[dict]:
        """いま読み込んでいる資料の一覧（画面に出す）。"""
        return [{"name": path.name,
                 "chars": len(path.read_text(encoding="utf-8", errors="replace"))}
                for path in sorted(self.directory().glob("*"))
                if path.is_file() and path.suffix.lower() in PREP_SUFFIXES]

    def status(self) -> dict:
        """資料の一覧と、そのうち要約に渡っている文字数。

        「渡っている文字数」を必ず出す。全文が入るとは限らないのに黙って切られると、
        資料を付けたのに効いていない理由が分からなくなる。
        """
        loaded = 0 if (not self.text or self.text.startswith("（前回タスク")) else len(self.text)
        passed = self.for_prompt or self.text
        digested = bool(loaded) and passed is not self.text and len(passed) < loaded
        return {
            "files": self.files(),
            "chars": loaded,
            "used_chars": min(len(passed) if loaded else 0, self.limit),
            "limit": self.limit,
            "digested": digested,
        }

    # -------------------------------------------------------------- 読み込み

    def load(self, prep_dir: Path | str | None = None) -> str:
        """資料を読み直して全文を作る（まとめは作らない）。"""
        path = Path(prep_dir) if prep_dir is not None else self.directory()
        parts: list[str] = []
        if path.exists():
            # .md 以外も読む（画面から足せる種類に合わせる）。並びは名前順で毎回同じにする
            for file in sorted(path.iterdir()):
                if not file.is_file() or file.suffix.lower() not in PREP_SUFFIXES:
                    continue
                content = file.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    # どれがどの資料かが分かるように名前を見出しにする（複数を足せるため）
                    parts.append(f"### {file.stem}\n{content}")
                    logger.info("Loaded prep file: %s", file.name)
        self.text = "\n\n".join(parts) if parts else EMPTY
        return self.text

    def reload(self, note: str = "") -> dict:
        """資料を読み直し、長ければ 1 回だけまとめて、変わったことを知らせる。"""
        self.load()
        self.for_prompt, digest_note = self.compress()
        if self._on_change is not None:
            self._on_change()
        status = {**self.status(), "note": "。".join(part for part in (note, digest_note) if part)}
        if self._publish is not None:
            self._publish("prep", status)
        return status

    # ---------------------------------------------------------------- 出し入れ

    def attach_file(self, name: str, data: bytes) -> dict:
        """**ファイル**を足す（PDF・Word・Excel・PowerPoint も）。

        本文はここ（手元）で取り出す。元のファイルもセッションに残す。
        取り出せるのは**文字だけ**。スキャンしただけの PDF は断る（黙って空を読み込まない）。
        """
        if len(data) > PREP_MAX_BYTES:
            raise ValueError(f"ファイルが大きすぎます（{len(data) / 1024 / 1024:.0f} MB・"
                             f"上限 {PREP_MAX_BYTES // 1024 // 1024} MB）")
        safe = safe_name(name)
        try:
            text = extract_text(safe, data)
        except UnreadableDocument as error:
            raise ValueError(str(error)) from error
        if Path(safe).suffix.lower() in DOCUMENT_SUFFIXES:
            originals = self.directory() / PREP_ORIGINALS
            originals.mkdir(exist_ok=True)
            (originals / safe).write_bytes(data)
            safe = f"{Path(safe).stem}.md"      # 取り出した本文は .md として置く
            logger.info("資料から本文を取り出しました: %s（%d 文字）", name, len(text))
        return self.attach(safe, text)

    def attach(self, name: str, text: str) -> dict:
        """文字の資料を足す。会議の途中でも足せる（次の状態更新から効く）。"""
        text = str(text)
        if not text.strip():
            raise ValueError("中身が空です")
        if len(text) > PREP_MAX_CHARS:
            raise ValueError(f"大きすぎます（{len(text):,} 文字・上限 {PREP_MAX_CHARS:,} 文字）")
        safe = safe_name(name)
        suffix = Path(safe).suffix.lower()
        if suffix in DOCUMENT_SUFFIXES:
            # PDF・Word・Excel・PowerPoint は本文を取り出してから来る（`attach_file`）
            raise ValueError(f"{suffix} はファイルとして足してください（本文を取り出してから読み込みます）")
        if suffix not in TEXT_SUFFIXES:
            raise ValueError(f"読めない種類です（{'・'.join(sorted(PREP_SUFFIXES))} のどれかにしてください）")
        (self.directory() / safe).write_text(text, encoding="utf-8")
        logger.info("事前資料を足しました: %s（%d 文字）", safe, len(text))
        return self.reload(note=f"{safe} を足しました")

    def remove(self, name: str) -> dict:
        """足した資料を外す（間違えて付けたとき）。"""
        path = self.directory() / safe_name(name)
        if not path.exists():
            raise KeyError(f"その資料はありません: {name}")
        path.unlink()
        logger.info("事前資料を外しました: %s", path.name)
        return self.reload(note=f"{path.name} を外しました")

    # ------------------------------------------------------------------ まとめ

    def compress(self) -> tuple[str, str]:
        """長い資料を「会議中ずっと載せる手元メモ」に 1 回だけ圧縮する。

        まとめに失敗したら、黙って切り詰めた全文へ落ちる（会議は止めない）。
        送り先は、そのとき使っている要約のエンジン。承認前は必ず手元。
        """
        text = self.text
        limit = self.limit
        self.digest_terms = []
        if not text or text.startswith("（前回タスク") or len(text) <= limit:
            return text, ""
        prompt_path = self.prompts_dir / "prep_digest.md"
        if not prompt_path.exists():
            return text, "資料が長いので先頭だけ使います"
        # 手元のモデルは文脈が狭い。渡しすぎると黙って頭から捨てられるので、こちらで切る
        max_input = PREP_DIGEST_MAX_INPUT if self._engine() == "gemini" else PREP_DIGEST_LOCAL_INPUT
        read_chars = min(len(text), max_input)
        try:
            data = self._llm().chat_json(
                prompt_path.read_text(encoding="utf-8"),
                text[:max_input],
                PREP_DIGEST_SCHEMA,
                num_ctx=PREP_DIGEST_NUM_CTX,
                num_predict=PREP_DIGEST_NUM_PREDICT,
                think=False,
            )
        except Exception as error:  # noqa: BLE001 — 会議は止めない
            logger.warning("資料のまとめに失敗しました（先頭だけ使います）: %s", error)
            return text, f"資料が長いのでまとめようとしましたが失敗しました（先頭 {limit:,} 文字を使います）"
        digest = render_digest(data, len(text), limit)
        self.digest_terms = [str(word).strip() for word in (data.get("terms") or []) if str(word).strip()]
        logger.info("資料をまとめました: %d 文字（うち %d 文字を読んだ）→ %d 文字",
                    len(text), read_chars, len(digest))
        note = f"長いので 1 回だけまとめました（{len(text):,} 文字 → {len(digest):,} 文字）"
        if read_chars < len(text):
            note += (f"。まとめに読めたのは先頭 {read_chars:,} 文字までです"
                     "（手元のモデルの文脈に収めるため。全部読ませるなら要約を外で回す）")
        return digest, note
