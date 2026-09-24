"""Ollama ローカル LLM クライアント。"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LlmConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gemma4"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_sec: int = 120
    engine: str = "local"
    """会議中の要約をどこで回すか。`local`（Ollama）| `gemini`。Ollama 自身は使わないが、
    `settings.yaml` の `llm:` をそのまま渡す呼び出し側があるので受け側で受ける（下の gemini と同じ理由）。
    読むのは `src/meeting_orchestrator.py`。"""

    gemini: dict = field(default_factory=dict)
    """会議中の要約を外（Gemini）で回すときの設定。Ollama は使わないが、`settings.yaml` の
    `llm:` をそのまま `LlmConfig(**config["llm"])` に渡す呼び出し側があるので、**受け側で受ける**
    （下の final_pass と同じ理由。2026-09-09 に同じ形で一度落ちている）。中身は
    `src.llm.gemini_client.GeminiLlmConfig` が読む。"""

    final_pass: str = "none"
    """"none" | "claude_cli"。Ollama は使わないが、settings の llm: セクションをそのまま受けられるように持つ
    （2026-09-09: settings に足したこのキーで、対面モードの src/orchestrator.py（不可侵）と replay が
    `LlmConfig(**config["llm"])` で落ちた。呼び出し側で pop するのではなく、受け側で受ける）。"""


@dataclass
class SummaryResponse:
    """Ollama からの要約レスポンスを構造化したもの。"""
    overall_summary: str = ""
    item_progress: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    chunk_summary: str = ""
    raw: str = ""


class OllamaClient:
    """Ollama /api/chat を使った要約クライアント。"""

    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_sec,
        )

    def unload(self) -> None:
        """モデルをメモリから降ろす（keep_alive=0）。失敗しても例外は出さない。"""
        try:
            self._client.post("/api/generate", json={"model": self.config.model, "keep_alive": 0})
            logger.warning("Ollama のモデル %s を降ろしました（次の呼び出しで読み直されます）", self.config.model)
        except httpx.HTTPError:
            logger.warning("Ollama のモデルを降ろせませんでした", exc_info=True)

    def health_check(self) -> bool:
        """Ollama が起動しているか確認。"""
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def summarize(
        self,
        system_prompt: str,
        hearing_items: str,
        overall_summary_so_far: str,
        recent_transcript: str,
    ) -> SummaryResponse:
        """直近の会話テキストを要約して構造化レスポンスを返す。"""
        user_content = (
            f"## ヒアリング項目\n{hearing_items}\n\n"
            f"## これまでの全体要約\n{overall_summary_so_far or '（まだありません）'}\n\n"
            f"## 直近5分間の会話テキスト\n{recent_transcript}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.info("Ollama へ要約リクエスト送信 (model=%s)", self.config.model)

        resp = self._client.post(
            "/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        resp.raise_for_status()

        content = resp.json()["message"]["content"]
        logger.debug("Ollama raw response length: %d chars", len(content))

        return self._parse_response(content)

    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        num_ctx: int,
        num_predict: int,
        think: bool | None = None,
    ) -> dict:
        """JSON Schema 制約付きの chat 応答を辞書として返す。"""
        logger.info("Ollama へ JSON リクエスト送信 (model=%s)", self.config.model)
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"temperature": self.config.temperature, "num_ctx": num_ctx, "num_predict": num_predict},
        }
        if think is not None:
            body["think"] = think
        resp = self._client.post(
            "/api/chat",
            json=body,
        )
        resp.raise_for_status()
        payload = resp.json()
        content = (payload.get("message") or {}).get("content", "")
        logger.debug("Ollama JSON response length: %d chars", len(content))
        if payload.get("error") or not content.strip():
            # GPU リセットのあと、Ollama は計算に失敗しても HTTP 200 で空の応答を返し続ける
            #   （2026-09-11 本番: 11:10〜11:22 の 12 分間ずっと）。モデルを降ろすと次の呼び出しで
            #   作り直されて直るので、ここで降ろしてから失敗を返す（呼び出し側の再試行で読み直される）
            self.unload()
            raise ValueError(
                f"Ollama が空の応答を返しました（error={payload.get('error')!r}, done_reason={payload.get('done_reason')!r}）。"
                "モデルを降ろしたので、次の試行で読み直します"
            )
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama の JSON 応答を読めません") from exc
        if not isinstance(result, dict):
            raise ValueError("Ollama の JSON 応答がオブジェクトではありません")
        return result

    def generate_text(self, system: str, user: str, *, num_ctx: int = 32768, num_predict: int = 4000,
                      timeout_sec: float | None = None) -> str:
        """形の決まっていない文章（議事録など）を返す。JSON にしない分、長い出力に向く。"""
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "options": {"temperature": self.config.temperature, "num_ctx": num_ctx, "num_predict": num_predict},
        }
        resp = self._client.post("/api/chat", json=body, timeout=timeout_sec or self.config.timeout_sec)
        resp.raise_for_status()
        content = ((resp.json().get("message") or {}).get("content") or "").strip()
        if not content:
            raise ValueError("Ollama が空の文章を返しました")
        return content

    @staticmethod
    def _parse_response(content: str) -> SummaryResponse:
        """セクション区切りのレスポンスをパースする。"""
        result = SummaryResponse(raw=content)

        sections = {
            "OVERALL_SUMMARY": "",
            "ITEM_PROGRESS": "",
            "SUGGESTIONS": "",
            "CHUNK_SUMMARY": "",
        }

        current_section = None
        lines = content.split("\n")

        for line in lines:
            section_match = re.search(r"SECTION:\s*(\w+)", line)
            if section_match:
                key = section_match.group(1)
                if key in sections:
                    current_section = key
                    continue
            if current_section:
                sections[current_section] += line + "\n"

        result.overall_summary = sections["OVERALL_SUMMARY"].strip()
        result.chunk_summary = sections["CHUNK_SUMMARY"].strip()

        # ITEM_PROGRESS: 行ごとにリストへ
        progress_text = sections["ITEM_PROGRESS"].strip()
        if progress_text:
            result.item_progress = [
                line.strip() for line in progress_text.split("\n")
                if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))
            ]

        # SUGGESTIONS: 行ごとにリストへ
        suggestions_text = sections["SUGGESTIONS"].strip()
        if suggestions_text:
            result.suggestions = [
                line.strip() for line in suggestions_text.split("\n")
                if line.strip() and (line.strip().startswith("-") or line.strip().startswith("*"))
            ]

        # パースに失敗した場合はフォールバック
        if not result.overall_summary and not result.chunk_summary:
            logger.warning("セクションパースに失敗。生レスポンスをchunk_summaryに格納")
            result.chunk_summary = content.strip()

        return result

    def close(self) -> None:
        self._client.close()
