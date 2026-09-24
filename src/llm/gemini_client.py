"""Gemini を使った JSON 応答クライアント（`OllamaClient` と同じ口）。

会議中の要約・状態更新・ファクトチェックを、手元の Ollama ではなく外の API で回すための差し替え。
必要な口は `chat_json` ひとつ（`state_updater.py` と `factcheck.py` が使うのはこれだけ）。

なぜ差し替えたいか（運用者 2026-09-13）: **マシンスペックへの依存を下げたい**。
このシステムは他者への販売や OSS 公開も視野に入っている。Ollama（gemma4）と mlx-whisper は
どちらも大きなモデルの取得と強い GPU を前提にするため、配れる範囲がとても狭い。

もともとローカルにしていた理由は費用ではなく**機密性**だった
（`system-outline.md`「処理をすべてオフラインで完結させることで、機密性の高いインタビューでも
安全に運用できる構成」）。ただし音声そのものを外へ出す構成にした時点で、その前提は既に崩れている
— 文字起こしは音声から作られるので、要約だけ手元に置いても守れるものは残らない。

送るのは**テキストだけ**（音声は送らない）。課金の確認は呼び出し側の関門で行う。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class GeminiLlmConfig:
    """`settings.yaml` の `llm.gemini`。"""

    model: str = "gemini-3.5-flash"
    key_file: str = ""
    temperature: float = 0.3
    timeout_sec: int = 120
    thinking_budget: int = 0
    """思考トークンは切る。要約に推論は要らないのに、入れたままだと出力の費用が跳ねる
    （`gemini_transcribe.py` の冒頭: 本文 621 に対し思考 2,332＝4 倍）。"""


@dataclass
class Usage:
    """使った量（費用の見積もりに使う）。"""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    failures: int = 0


class GeminiClient:
    """`generateContent` に JSON Schema を付けて呼ぶ。`OllamaClient` と同じ `chat_json` を持つ。"""

    def __init__(self, config: GeminiLlmConfig) -> None:
        self.config = config
        self.usage = Usage()
        key = ""
        if config.key_file:
            path = Path(config.key_file).expanduser()
            if path.exists():
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                key = lines[0].strip() if lines else ""
        self._key = key or os.environ.get("GEMINI_API_KEY", "")
        if not self._key:
            raise ValueError("Gemini の API キーがありません（llm.gemini.key_file か GEMINI_API_KEY）")
        self._client = httpx.Client(base_url=BASE, timeout=config.timeout_sec)

    # ---------------------------------------------------------------- 同じ口

    def health_check(self) -> bool:
        """キーが通り、モデルが見えるか。"""
        try:
            response = self._client.get(f"/models/{self.config.model}",
                                        headers={"x-goog-api-key": self._key})
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        num_ctx: int = 0,
        num_predict: int = 0,
        think: bool | None = None,
    ) -> dict:
        """JSON Schema 制約つきの応答を辞書で返す。

        `num_ctx` は Gemini では意味を持たない（受けるだけ）。`num_predict` は出力の上限に使う。
        """
        body: dict = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "responseMimeType": "application/json",
                "responseSchema": _to_gemini_schema(schema),
                "thinkingConfig": {"thinkingBudget": self.config.thinking_budget},
            },
        }
        if num_predict:
            body["generationConfig"]["maxOutputTokens"] = num_predict

        started = time.monotonic()
        response = self._client.post(
            f"/models/{self.config.model}:generateContent",
            headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
            json=body,
        )
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            self.usage.failures += 1
            raise ValueError(f"Gemini が {response.status_code} を返しました: {response.text[:200]}")

        payload = response.json()
        usage = payload.get("usageMetadata", {})
        self.usage.calls += 1
        self.usage.prompt_tokens += int(usage.get("promptTokenCount", 0))
        self.usage.output_tokens += int(usage.get("candidatesTokenCount", 0))
        self.usage.latencies.append(elapsed)

        text = ""
        for candidate in payload.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                text += part.get("text", "") or ""
        if not text.strip():
            self.usage.failures += 1
            raise ValueError(f"Gemini が空の応答を返しました（{payload.get('promptFeedback')}）")
        try:
            result = json.loads(text)
        except json.JSONDecodeError as error:
            self.usage.failures += 1
            raise ValueError("Gemini の JSON 応答を読めません") from error
        if not isinstance(result, dict):
            raise ValueError("Gemini の JSON 応答がオブジェクトではありません")
        return result

    def generate_text(self, system: str, user: str, *, max_tokens: int = 8000) -> str:
        """形の決まっていない文章（議事録など）を返す。送るのはテキストだけ。"""
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": self.config.thinking_budget},
            },
        }
        payload = self._post(body)
        return _text_of(payload)

    def search_json(self, prompt: str, *, thinking_budget: int | None = None) -> tuple[str, list[dict]]:
        """Google 検索つきで答えさせ、(本文, 出典) を返す。

        `google_search` を**明示的に**付けたときだけ検索する（付けなければ記憶で答える）。
        出典は `groundingMetadata` から取る＝**検索で実際に使われたページ**（モデルが書いた URL ではない）。
        費用: 月 5,000 回までは無料、超えると 1,000 回 $14（2026-09-14 時点の価格表）。
        """
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.2,
                                 "thinkingConfig": {"thinkingBudget": self.config.thinking_budget
                                                    if thinking_budget is None else thinking_budget}},
        }
        payload = self._post(body)
        sources: list[dict] = []
        for candidate in payload.get("candidates", []):
            metadata = candidate.get("groundingMetadata") or {}
            for chunk in metadata.get("groundingChunks", []) or []:
                web = chunk.get("web") or {}
                if web.get("uri"):
                    sources.append({"uri": web["uri"], "title": web.get("title", "")})
        return _text_of(payload), sources

    def _post(self, body: dict) -> dict:
        started = time.monotonic()
        response = self._client.post(
            f"/models/{self.config.model}:generateContent",
            headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
            json=body,
        )
        self.usage.latencies.append(time.monotonic() - started)
        if response.status_code != 200:
            self.usage.failures += 1
            raise ValueError(f"Gemini が {response.status_code} を返しました: {response.text[:200]}")
        payload = response.json()
        usage = payload.get("usageMetadata", {})
        self.usage.calls += 1
        self.usage.prompt_tokens += int(usage.get("promptTokenCount", 0))
        self.usage.output_tokens += int(usage.get("candidatesTokenCount", 0))
        return payload

    def unload(self) -> None:
        """外の API なので降ろすものは無い（口を揃えるためだけに置く）。"""

    def close(self) -> None:
        self._client.close()


def _text_of(payload: dict) -> str:
    text = ""
    for candidate in payload.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            text += part.get("text", "") or ""
    if not text.strip():
        raise ValueError(f"Gemini が空の応答を返しました（{payload.get('promptFeedback')}）")
    return text.strip()


def _to_gemini_schema(schema: dict) -> dict:
    """Ollama 向けの JSON Schema を Gemini の responseSchema に直す。

    Gemini は `additionalProperties` を受け付けない（400）。型名も大文字を期待する。
    """
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for key, value in schema.items():
        if key in {"additionalProperties", "$schema", "title", "default"}:
            continue
        if key == "type" and isinstance(value, str):
            out["type"] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {name: _to_gemini_schema(item) for name, item in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        else:
            out[key] = value
    return out
