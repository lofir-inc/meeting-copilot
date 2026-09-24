"""Ollama クライアントのパース処理テスト（API接続不要）。"""

from src.llm.ollama_client import LlmConfig, OllamaClient, SummaryResponse


class FakeResponse:
    """JSON チャットのテスト用レスポンス。"""

    def raise_for_status(self) -> None:
        """成功レスポンスとして扱う。"""

    def json(self) -> dict:
        """空の JSON オブジェクト応答を返す。"""
        return {"message": {"content": "{}"}}


class FakeHttpClient:
    """リクエスト本文を記録するテスト用 HTTP クライアント。"""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def post(self, path: str, json: dict) -> FakeResponse:
        """送信先と JSON 本文を記録する。"""
        assert path == "/api/chat"
        self.bodies.append(json)
        return FakeResponse()


def test_chat_json_adds_think_only_when_explicit() -> None:
    """think は指定された場合だけ Ollama のリクエスト本文へ入る。"""
    client = object.__new__(OllamaClient)
    client.config = LlmConfig()
    client._client = FakeHttpClient()
    client.chat_json("system", "user", {}, num_ctx=1, num_predict=2)
    client.chat_json("system", "user", {}, num_ctx=1, num_predict=2, think=False)
    assert "think" not in client._client.bodies[0]
    assert client._client.bodies[1]["think"] is False


class TestParseResponse:
    def test_parse_well_formed_response(self):
        content = """
### SECTION: OVERALL_SUMMARY
インタビュー全体の要約テスト。重要な決定事項がありました。

### SECTION: ITEM_PROGRESS
- ✅ 項目A: カバー済み
- ⚠️ 項目B: まだ確認できていない
- ✅ 項目C: 詳細に議論された

### SECTION: SUGGESTIONS
- 項目Bについて質問を追加すべき
- 予算に関する深掘りが必要

### SECTION: CHUNK_SUMMARY
トピック: プロジェクト体制について
- Interviewer: 体制変更の背景を質問
- Guest: 来月からチーム拡大予定と回答
"""
        result = OllamaClient._parse_response(content)

        assert "要約テスト" in result.overall_summary
        assert len(result.item_progress) == 3
        assert "✅" in result.item_progress[0]
        assert len(result.suggestions) == 2
        assert "トピック" in result.chunk_summary

    def test_parse_empty_sections(self):
        content = """
### SECTION: OVERALL_SUMMARY

### SECTION: ITEM_PROGRESS

### SECTION: SUGGESTIONS

### SECTION: CHUNK_SUMMARY
"""
        result = OllamaClient._parse_response(content)
        assert result.overall_summary == ""
        assert result.item_progress == []
        assert result.suggestions == []

    def test_parse_fallback_on_no_sections(self):
        content = "これはセクション区切りのないフリーテキストです。"
        result = OllamaClient._parse_response(content)
        # フォールバック: chunk_summary に全文が入る
        assert "フリーテキスト" in result.chunk_summary
        assert result.raw == content
