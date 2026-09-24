"""Ollama クライアントのテスト（HTTP はモック）。"""

import httpx
import pytest

from src.llm.ollama_client import LlmConfig, OllamaClient


def _client(handler) -> OllamaClient:
    client = OllamaClient(LlmConfig())
    client._client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return client


def test_empty_response_unloads_the_model_before_failing():
    """2026-09-11 本番: GPU リセット後の Ollama は HTTP 200 で空の応答を返し続けた。降ろすと直る。"""
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": ""}, "done": True})
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    with pytest.raises(ValueError, match="空の応答"):
        _client(handler).chat_json("system", "user", {}, num_ctx=8192, num_predict=10)
    assert seen[-1] == ("/api/generate", {"model": "gemma4", "keep_alive": 0})


def test_valid_json_is_returned():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"ok": true}'}, "done": True})

    assert _client(handler).chat_json("system", "user", {}, num_ctx=8192, num_predict=10) == {"ok": True}
