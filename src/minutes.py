"""会議のあとに、議事録（`minutes.md`）まで作る。**使える手段から順に**選ぶ。

誰の環境でも最後まで行けるように、手段を 4 段で持つ（2026-09-14 運用者 依頼「Claude のサブスクが
無い人でも議事録まで行ける逃げ道を」）。

| 段 | 手段 | 要るもの | 外へ出るか |
|---|---|---|---|
| 1 | Claude CLI | Claude のサブスク（`claude` コマンド） | テキストだけ（従量課金なし） |
| 2 | Gemini | API キー＋課金が有効なプロジェクト＋**その会議の承認** | テキストだけ（従量） |
| 3 | 手元の LLM | Ollama | 出ない |
| 4 | 貼り付け用の指示書 | 何も要らない | 出ない（人が好きなチャットに貼る） |

`auto` は 1 → 2 → 3 → 4 の順に、**使えるものを**選ぶ。途中で失敗したら次へ落ちる。

Gemini は**その会議で外へのテキスト送信が承認されているとき**だけ候補にする。承認なしに、
会議の全文を黙って外へ出すことはしない（`src/stt/external_consent.py` と同じ考え）。

手元の LLM は文脈が狭い（数万字の全文が 1 回に入らない）。長いときは**分けて要点を取り、
最後にまとめる**（map → reduce）。精度はクラウドより落ちるが、オフラインで完結する。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.known_people import roster_block
from src.text.glossary_candidates import LLM_FILE, split_minutes

logger = logging.getLogger(__name__)

MINUTES_FILE = "minutes.md"
PROMPT_FILE = "minutes_prompt.md"
INPUT_FILE = "minutes_input.md"
ENGINES = ("claude_cli", "gemini", "local", "none")


@dataclass
class MinutesConfig:
    """`settings.yaml` の `minutes`。"""

    engine: str = "auto"
    """auto | claude_cli | gemini | local | none。auto は使えるものを上から選ぶ。"""

    local_chunk_chars: int = 12000
    """手元の LLM に 1 回で渡す全文の長さ。超えたら分けて要点を取り、最後にまとめる。"""

    timeout_sec: int = 600

    @classmethod
    def from_mapping(cls, values: dict | None) -> "MinutesConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (values or {}).items() if key in known})


@dataclass
class Availability:
    """いま使える手段。"""

    claude_cli: bool = False
    gemini: bool = False
    """「キーがある」だけでは True にしない。**その会議で外へのテキスト送信が承認された**ときだけ。"""
    local: bool = False


def resolve_order(config: MinutesConfig, available: Availability) -> list[str]:
    """試す順番を返す。最後は必ず none（貼り付け用の指示書）で終わる＝どこかで必ず止まる。"""
    if config.engine != "auto":
        chosen = [config.engine] if config.engine in ENGINES else []
        return chosen + (["none"] if chosen != ["none"] else [])
    order = [name for name in ("claude_cli", "gemini", "local") if getattr(available, name)]
    return order + ["none"]


def split_transcript(text: str, limit: int) -> list[str]:
    """全文を、行の切れ目で `limit` 字ほどに分ける（手元の LLM の文脈に収めるため）。"""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        if size + len(line) > limit and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


class MinutesMaker:
    """議事録の材料（`minutes_input.md`）から、議事録（`minutes.md`）を作る。"""

    def __init__(
        self,
        session_dir: Path,
        prompts_dir: Path,
        config: MinutesConfig | None = None,
        *,
        claude=None,
        gemini=None,
        local=None,
        on_progress: Callable[[str], None] | None = None,
        known_people: list[str] | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.prompts_dir = Path(prompts_dir)
        self.config = config or MinutesConfig()
        self.known_people = list(known_people or [])
        """その会社の会議に出てくる人。一覧に無い人名を「誤変換の候補」に挙げさせる。

        直させるためではない（`prompts/minutes.md` は人名を推測しない決まり）。
        気づかせるため。`src/known_people.py` の注記を読むこと。
        """
        self._claude = claude
        self._gemini = gemini
        self._local = local
        self._say = on_progress or (lambda message: logger.info("%s", message))

    def instructions(self) -> str:
        told = (self.prompts_dir / "minutes.md").read_text(encoding="utf-8")
        return told + roster_block(self.known_people)

    def material(self) -> str:
        path = self.session_dir / INPUT_FILE
        if not path.exists():
            raise FileNotFoundError(f"議事録の材料がありません: {path}")
        return path.read_text(encoding="utf-8")

    def make(self, order: list[str]) -> tuple[str, Path]:
        """順に試し、(使った手段, 書き出した先) を返す。どれも駄目なら指示書を書いて返す。"""
        text = self.material()
        for engine in order:
            if engine == "none":
                return "none", self._write_prompt(text)
            try:
                self._say(f"議事録を作っています（{_LABELS[engine]}）…")
                minutes = self._generate(engine, text)
            except Exception as error:  # noqa: BLE001 — 次の手段へ落ちる
                logger.warning("議事録を %s で作れませんでした: %s", engine, error)
                self._say(f"  {_LABELS[engine]} では作れませんでした（{str(error)[:80]}）。次の手段を試します")
                continue
            minutes, suspects = split_minutes(minutes)
            path = self.session_dir / MINUTES_FILE
            path.write_text(_clean(minutes).rstrip() + "\n", encoding="utf-8")
            # 誤変換の候補は議事録と同じ 1 回の呼び出しで取る（全文を読み直す呼び出しを増やさない）
            (self.session_dir / LLM_FILE).write_text(json.dumps(suspects, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
            return engine, path
        return "none", self._write_prompt(text)

    # ------------------------------------------------------------------ 手段ごと

    def _generate(self, engine: str, text: str) -> str:
        instructions = self.instructions()
        if engine == "claude_cli":
            if self._claude is None:
                raise RuntimeError("Claude CLI が使えません")
            return (self._claude.generate_text(f"{instructions}\n\n---\n\n{text}",
                                                     timeout_sec=self.config.timeout_sec))
        if engine == "gemini":
            if self._gemini is None:
                raise RuntimeError("Gemini が使えません（その会議で外へのテキスト送信が承認されていない）")
            return (self._gemini.generate_text(instructions, text))
        if engine == "local":
            if self._local is None:
                raise RuntimeError("手元の LLM が使えません")
            return (self._local_minutes(instructions, text))
        raise ValueError(f"知らない手段: {engine}")

    def _local_minutes(self, instructions: str, text: str) -> str:
        """手元の LLM で作る。長いときは分けて要点を取り（map）、最後にまとめる（reduce）。"""
        body, _, state = text.partition("## 最終状態")
        chunks = split_transcript(body, self.config.local_chunk_chars)
        if len(chunks) <= 1:
            return self._local.generate_text(instructions, text)
        notes: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            self._say(f"  手元の LLM で要点を取っています（{index}/{len(chunks)}）…")
            notes.append(self._local.generate_text(
                "会議の全文の一部を渡します。この部分で出た決定事項・TODO（担当と期限）・論点・数字と固有名詞を、"
                "書いてあることだけ箇条書きで抜き出してください。推測で補わないこと。前置きは書かないこと。",
                chunk, num_predict=1500))
        joined = "\n\n".join(f"### 部分 {index}\n{note}" for index, note in enumerate(notes, 1))
        return self._local.generate_text(
            instructions,
            f"（全文が長いので、部分ごとに要点を抜き出したものを渡します）\n\n{joined}\n\n## 最終状態{state}")

    def _write_prompt(self, text: str) -> Path:
        """最後の逃げ道: 指示書と材料を 1 つにして、どのチャット（ChatGPT・Claude・Gemini の画面）にも
        貼れる形で置く。何も入っていない環境でも、議事録までは人の手 1 回で行ける。"""
        path = self.session_dir / PROMPT_FILE
        path.write_text(
            "<!-- このファイルの中身をまるごと、お使いの AI チャットに貼り付けてください。 -->\n\n"
            f"{self.instructions()}\n\n---\n\n{text}", encoding="utf-8")
        return path


_LABELS = {"claude_cli": "Claude CLI", "gemini": "Gemini", "local": "手元の LLM", "none": "貼り付け用の指示書"}


def label(engine: str) -> str:
    return _LABELS.get(engine, engine)


def _clean(text: str) -> str:
    """前置き（「以下が議事録です」など）とコードブロックの囲いを外す。"""
    text = text.strip()
    fenced = re.search(r"```(?:markdown|md)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced and fenced.group(1).lstrip().startswith("#"):
        text = fenced.group(1).strip()
    # 見出しより前にある前置きの文を落とす（見出しが無ければ、そのまま返す）
    heading = re.search(r"^# ", text, re.MULTILINE)
    return text[heading.start():].strip() if heading else text
