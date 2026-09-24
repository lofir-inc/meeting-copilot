"""ファクトチェック（検出＋ローカル判定）のテスト。Ollama は使わない。"""

import pytest

from src.llm.factcheck import Claim, FactCheckConfig, FactChecker, strip_urls
from src.stt.whisper_client import TranscriptSegment


class FakeClient:
    """chat_json の戻り値を順に返す代役。"""

    def __init__(self, *values):
        self.values = list(values)
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system, user, schema, **kwargs):
        self.calls.append((system, user))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _window():
    return [TranscriptSegment("参加者A", "ゼロクリック検索で流入が 40% 下がったらしいですよ", 120.0, 125.0, "")]


def test_detected_claims_get_ids_and_the_window_time():
    checker = FactChecker(FakeClient({"claims": [{"quote": "流入が 40% 下がった", "kind": "stat", "speaker": "参加者A"}]}))
    claims = checker.detect(_window())
    assert [(c.id, c.kind, c.speaker, c.at) for c in claims] == [("C1", "stat", "参加者A", 125.0)]
    assert claims[0].verdict == "確認中"


def test_at_most_one_claim_per_window():
    """2026-09-12 の試行: 50 分で 71 件拾い、ほとんどが社内の話だった。1 回 1 件に絞る。"""
    many = {"claims": [{"quote": f"主張{i}", "kind": "stat", "speaker": "参加者A"} for i in range(5)]}
    assert len(FactChecker(FakeClient(many)).detect(_window())) == 1


def test_empty_window_does_not_call_the_model():
    client = FakeClient()
    assert FactChecker(client).detect([]) == []
    assert client.calls == []


def test_detection_failure_is_not_fatal():
    assert FactChecker(FakeClient(ValueError("bad json"))).detect(_window()) == []


def test_verify_keeps_only_known_verdicts():
    claim = Claim("C1", "流入が 40% 下がった", "stat", "参加者A", 125.0)
    checker = FactChecker(FakeClient({"verdict": "たぶん正しい", "note": "根拠"}))
    assert checker.verify(claim).verdict == "不明"


def test_verify_strips_urls_from_the_note():
    """段1はモデルの記憶で URL を書きがちだが、それは出典にならない。"""
    claim = Claim("C1", "流入が 40% 下がった", "stat", "参加者A", 125.0)
    checker = FactChecker(FakeClient({"verdict": "要確認", "note": "2024 年の調査では 30% https://example.com/report"}))
    verified = checker.verify(claim)
    assert verified.verdict == "要確認"
    assert "http" not in verified.note


def test_verify_off_leaves_the_claim_untouched():
    claim = Claim("C1", "流入が 40% 下がった", "stat", "参加者A", 125.0)
    client = FakeClient()
    assert FactChecker(client, FactCheckConfig(verify="off")).verify(claim).verdict == "確認中"
    assert client.calls == []


def test_verify_failure_marks_unknown():
    claim = Claim("C1", "流入が 40% 下がった", "stat", "参加者A", 125.0)
    assert FactChecker(FakeClient(ValueError("boom"))).verify(claim).verdict == "不明"


@pytest.mark.parametrize("text,expected", [
    ("2024 年の調査では 30% https://example.com/a", "2024 年の調査では 30%"),
    ("URL なしの根拠", "URL なしの根拠"),
])
def test_strip_urls(text, expected):
    assert strip_urls(text) == expected


class TestLoop:
    def test_flush_processes_what_was_pushed(self):
        from src.llm.factcheck import FactCheckLoop

        client = FakeClient({"claims": [{"quote": "流入が 40% 下がった", "kind": "stat", "speaker": "参加者A"}]},
                            {"verdict": "要確認", "note": "2024 年の数字とは違う"})
        seen: list[str] = []
        loop = FactCheckLoop(FactChecker(client), on_claim=lambda claim: seen.append(claim.id))
        for segment in _window():
            loop.push(segment)
        claims = loop.flush()
        assert [c.verdict for c in claims] == ["要確認"]
        assert seen == ["C1"]

    def test_flush_without_anything_pushed_is_quiet(self):
        from src.llm.factcheck import FactCheckLoop

        client = FakeClient()
        assert FactCheckLoop(FactChecker(client)).flush() == []
        assert client.calls == []
