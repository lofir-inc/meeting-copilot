"""Claude CLI を最終的なテキスト状態整理に使うクライアント。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

from pathlib import Path

from src.llm.meeting_state import DELTA_SCHEMA, MeetingState, StateDelta
from src.llm.state_updater import _validate_delta

_FINAL_PASS_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "meeting_final_pass.md"
_VERIFY_PROMPT = Path(__file__).resolve().parent.parent.parent / "prompts" / "factcheck_verify.md"

VERDICTS = ("一致", "要確認", "不明")

SEARCH_TOOLS = "WebSearch WebFetch"
"""裏取りで使わせる道具。検索して**実際に開く**ところまでやらせる（記憶の URL は出典にならない）。"""


def final_pass_prompt() -> str:
    """最終整理の指示文を返す。

    会議中の指示文（30〜60 秒の窓から差分だけ）を最終パスに使い回すと、全文を渡しても
    「新しい差分はない」と返ってくる。2026-09-12 に実測: 決定事項 0 件のまま変化なしだったが、
    全文から洗い出させると 8 件あった。
    """
    return _FINAL_PASS_PROMPT.read_text(encoding="utf-8")


def verify_prompt() -> str:
    """裏取りの指示文を返す。"""
    return _VERIFY_PROMPT.read_text(encoding="utf-8")


def verify_request(quote: str, context: str = "") -> str:
    """裏取り 1 件ぶんの指示文。Claude CLI と Gemini（検索つき）で同じものを使う。"""
    return (
        f"{verify_prompt()}\n\n## 裏取りする発言\n{quote.strip()}\n"
        + (f"\n## 前後の発言（文脈のためだけ。裏取りの対象ではない）\n{context.strip()}\n" if context.strip() else "")
        + '\n## 出力（この JSON だけ。前置きもコードブロックも書かない）\n'
        '{"verdict": "一致|要確認|不明", "note": "120 文字以内", "sources": ["実際に開いた URL"]}\n'
    )


def extract_json(text: str) -> dict:
    """返答の中から JSON オブジェクトを取り出す。

    Claude CLI は前置きの文やコードブロック（```json）を付けて返すことがある
    （2026-09-12 に最終パスで実際に失敗した）。素直に json.loads すると落ちる。
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = stripped.find("{"), stripped.rfind("}")
        candidate = stripped[start : end + 1] if start >= 0 and end > start else None
    if candidate is None:
        raise ValueError("Claude CLI の応答から JSON を取り出せませんでした")
    return json.loads(candidate)


class ClaudeCliClient:
    """音声を送らず、テキストだけを Claude CLI へ渡す。

    会議の最終パス（全文の読み直し）と、**発言 1 件の裏取り**（Web 検索つき）の 2 つに使う。
    どちらもサブスクの範囲で動き、従量課金は増えない。裏取りに Gemini API を使わない理由:
    ①API は `google_search` を明示的に有効にしないと**検索しない**（有効にすると 5,000 回/月の
    無料枠を超えたところから $14/1,000 回）②Claude CLI は検索して**実際にページを開ける**
    ③このシステムは既に最終パスで Claude CLI を使っており、契約の整理も済んでいる。
    """

    def __init__(self, model: str = "", timeout_sec: int = 300) -> None:
        self.model = model
        self.timeout_sec = timeout_sec

    def available(self) -> bool:
        """Claude CLI が PATH 上にあるかを返す。"""
        return shutil.which("claude") is not None

    def final_pass(self, state: MeetingState, transcript_text: str, system_prompt: str) -> StateDelta:
        """最終整理用の差分を Claude CLI から取得して検証する。"""
        command = ["claude", "-p", "--output-format", "json"]
        if self.model:
            command.extend(["--model", self.model])
        prompt = f"{system_prompt}\n\n## 現在の状態\n{state.to_prompt_text()}\n\n## 会議全文\n{transcript_text}\n\n{DELTA_SCHEMA}"
        result = subprocess.run(command, input=prompt, text=True, capture_output=True, timeout=self.timeout_sec, shell=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Claude CLI が失敗しました")
        data = json.loads(result.stdout)
        if isinstance(data, dict) and isinstance(data.get("result"), str):
            data = extract_json(data["result"])
        return _validate_delta(data)

    def generate_text(self, prompt: str, *, timeout_sec: int | None = None) -> str:
        """形の決まっていない文章（議事録など）を返す。サブスクの範囲で動く（従量課金なし）。"""
        command = ["claude", "-p", "--output-format", "json"]
        if self.model:
            command.extend(["--model", self.model])
        result = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                timeout=timeout_sec or self.timeout_sec, shell=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Claude CLI が失敗しました")
        data = json.loads(result.stdout)
        text = data.get("result", "") if isinstance(data, dict) else ""
        if not str(text).strip():
            raise ValueError("Claude CLI が空の文章を返しました")
        return str(text).strip()

    def verify(self, quote: str, context: str = "", *, timeout_sec: int | None = None) -> dict:
        """発言 1 件を Web で裏取りして `{"verdict", "note", "sources"}` を返す。

        送るのは**その発言と前後の文字起こしだけ**（音声は送らない）。
        実測（2026-09-14）: 1 件あたり 22 秒・検索 1 回で、出典に実際のページ URL が返る。
        """
        if not quote.strip():
            raise ValueError("発言が空です")
        command = ["claude", "-p", "--output-format", "json", "--allowedTools", SEARCH_TOOLS]
        if self.model:
            command.extend(["--model", self.model])
        prompt = verify_request(quote, context)
        result = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                timeout=timeout_sec or self.timeout_sec, shell=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Claude CLI が失敗しました")
        data = json.loads(result.stdout)
        if isinstance(data, dict) and isinstance(data.get("result"), str):
            data = extract_json(data["result"])
        return validate_verification(data)


def validate_verification(data: dict) -> dict:
    """返ってきた裏取りの形を確かめる。判定の語彙を勝手に増やさせない。"""
    if not isinstance(data, dict):
        raise ValueError("裏取りの応答がオブジェクトではありません")
    verdict = str(data.get("verdict", "")).strip()
    if verdict not in VERDICTS:
        raise ValueError(f"知らない判定です: {verdict!r}")
    sources = [str(url).strip() for url in (data.get("sources") or []) if str(url).strip()]
    # 出典に使えるのは http(s) で始まるものだけ（「公式サイト」のような言葉は出典にならない）
    sources = [url for url in sources if url.startswith("http")]
    return {"verdict": verdict, "note": str(data.get("note", "")).strip()[:200], "sources": sources[:5]}
