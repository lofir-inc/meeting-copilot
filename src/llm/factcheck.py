"""ファクトチェック — 裏取りの価値がある発言を拾い、ローカルで一次判定する（Phase 7a・7b 段1）。

会議では「◯年の調査で」「ニュースで見た」「◯◯が先日リリースされた」のような、確かめたい発言が混ざる。
ここでは **拾う（検出）** と **ローカルでの一次判定** までを行う。Web で一次ソースを引く段2は、
画面から頼まれたときだけ動かす想定（PLAN-meeting-v2.md の Phase 7）。

2026-09-12 の実測（2026-09-11 の会議 50 分ぶんに gemma4 で当てた）:
  基準が緩い版 71 件 → 厳しくして 25 件 → 「確かめ方」を書かせて 24 件。**どの版も大半が社内の案件・見積・
  自社の数字**で、外部で確かめられる主張は数件だった。8B では「会議の外の事実か」の線引きが安定しない。
  ∴ **自動検出は既定 off のまま**。実用は「気になった行を人が指して調べる」オンデマンド方式（Phase 7 の第一候補）。

守ること:
- 段1は **URL を出さない**。モデルの記憶の URL は出典にならない（実測で 29 本中 7 本が 404）。
- モデルが知らない新しい出来事は「不明」にする。知らないものを否定させない。
- 拾いすぎない。1 回の呼び出しで 0〜1 件（2026-09-12 の試行では 50 分で 71 件拾ってしまい、ほぼ社内の話だった）。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from src.llm.ollama_client import OllamaClient
from src.stt.whisper_client import TranscriptSegment

logger = logging.getLogger(__name__)

KINDS = ["stat", "date", "event", "product", "definition", "hearsay"]
VERDICTS = ["一致", "要確認", "不明"]

DETECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote", "kind", "speaker", "how_to_check"],
                "properties": {
                    "quote": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "speaker": {"type": "string"},
                    "how_to_check": {"type": "string"},
                },
            },
        }
    },
}

VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "note"],
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "note": {"type": "string"},
    },
}

DETECT_SYSTEM = """あなたは会議の書き起こしから「あとで裏を取る価値がある発言」だけを拾う担当です。

判断の基準はひとつだけ: **その会議にいない第三者が、検索や公開資料で確かめられるか。**

拾うもの（確かめられる・会議の外の事実）:
- stat: 公開された調査・統計の数字（「ゼロクリック検索で流入が 40% 下がった」）
- date: 公になっている出来事の時期（「2024 年の調査で」「先月の法改正で」）
- event: ニュースになった出来事（「◯◯社が買収された」）
- product: 一般に売られている製品・サービスの仕様・価格・提供状況（「◯◯は月 20 ドル」「無料枠がある」）
- definition: 一般的な用語の定義（「AGI というのは〜」）
- hearsay: 上のどれかを伝聞として言ったもの（「ニュースで見た」「サイトに書いてあった」）

拾わないもの（確かめられない・この会議の内側の話）:
- 自社・相手社の案件、見積、金額、スケジュール、進捗、受注や問い合わせの件数
- 個人の予定・作業状況・体調・社内の人の話
- 意見・感想・提案・方針・これからやること（「〜した方がいい」「〜しましょう」）
- 一般論として正しいかを論じても意味がない発言

**ほとんどの区間では 0 件が正解です。** 10 回のうち 8 回は空の配列を返すつもりで見てください。
1 回につき多くても 1 件。会議 1 時間で 2〜3 件が適量です。

how_to_check には「第三者がどう確かめるか」を具体的に書いてください（例:「公式サイトの料金ページで月額を確認」
「総務省の◯◯調査の該当年の数字を確認」）。**ここに書けないものは拾ってはいけません。**
「本人に聞く」「社内で確認する」しか書けないなら、それは会議の内側の話なので拾いません。
quote は発言そのままを 60 字以内で。speaker は発言者名をそのまま写してください。"""


VERIFY_SYSTEM = """あなたは会議で出た主張を、自分の知識だけで一次判定する担当です。

- 一致: 自分の知識とはっきり一致する
- 要確認: 自分の知識と食い違う、または古い可能性がある
- 不明: 知らない。特に自分の知識より後の出来事・製品は必ず「不明」

URL は書かないでください。記憶にある URL は出典になりません。
知らないものを「間違い」と書かないでください。「不明」です。
note は 60 字以内の日本語で、判断の根拠を一言。数字の主張なら自分の知っている値を添えてください。"""


@dataclass
class FactCheckConfig:
    """ファクトチェックの設定（`settings.yaml` の `meeting.factcheck`）。"""

    enabled: bool = False
    interval_sec: float = 60.0
    max_prompt_chars: int = 4000
    num_ctx: int = 8192
    num_predict: int = 300
    think: bool = False
    verify: str = "local"
    """`local`（既定・ローカルのみ）| `off`（検出だけ）。Web で引く段2は画面からの依頼で別途。"""


@dataclass
class Claim:
    """裏取りの価値がある発言 1 件。"""

    id: str
    quote: str
    kind: str
    speaker: str
    at: float
    verdict: str = "確認中"
    note: str = ""
    how_to_check: str = ""
    """第三者がどう確かめるか（検出時に書かせる。書けないものは拾わせない）。"""
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "quote": self.quote, "kind": self.kind, "speaker": self.speaker,
                "at": round(self.at, 2), "verdict": self.verdict, "note": self.note,
                "how_to_check": self.how_to_check, "sources": list(self.sources)}


def _window_text(window: list[TranscriptSegment], limit: int) -> str:
    lines = [
        f"[{int(segment.start_time // 60)}:{int(segment.start_time % 60):02d}] {segment.speaker}: {segment.text}"
        for segment in sorted(window, key=lambda value: value.start_time)
    ]
    text = "\n".join(lines)
    return text[-limit:] if len(text) > limit else text


class FactChecker:
    """発話の窓から主張を拾い、ローカルで一次判定する。"""

    def __init__(self, client: OllamaClient, config: FactCheckConfig | None = None) -> None:
        self.client = client
        self.cfg = config or FactCheckConfig()
        self._count = 0

    def detect(self, window: list[TranscriptSegment]) -> list[Claim]:
        """窓から主張を拾う（0〜2 件）。失敗したら空。"""
        if not window:
            return []
        text = _window_text(window, self.cfg.max_prompt_chars)
        try:
            data = self.client.chat_json(
                DETECT_SYSTEM, f"## 直近の発話\n{text}", DETECT_SCHEMA,
                num_ctx=self.cfg.num_ctx, num_predict=self.cfg.num_predict, think=self.cfg.think,
            )
        except Exception as exc:  # noqa: BLE001 — 会議を止めない
            logger.warning("ファクトチェックの検出に失敗しました: %s", exc)
            return []
        claims: list[Claim] = []
        at = max((segment.end_time for segment in window), default=0.0)
        for item in data.get("claims", [])[:1]:
            quote = str(item.get("quote", "")).strip()
            if not quote:
                continue
            self._count += 1
            claims.append(Claim(id=f"C{self._count}", quote=quote[:60], kind=str(item.get("kind", "stat")),
                                speaker=str(item.get("speaker", "")).strip(), at=at,
                                how_to_check=str(item.get("how_to_check", "")).strip()[:80]))
        return claims

    def verify(self, claim: Claim) -> Claim:
        """ローカルの知識だけで一次判定する（URL は出さない）。"""
        if self.cfg.verify == "off":
            return claim
        try:
            data = self.client.chat_json(
                VERIFY_SYSTEM, f"## 主張\n{claim.quote}\n\n## 種別\n{claim.kind}", VERIFY_SCHEMA,
                num_ctx=self.cfg.num_ctx, num_predict=self.cfg.num_predict, think=self.cfg.think,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ファクトチェックの判定に失敗しました: %s", exc)
            claim.verdict = "不明"
            claim.note = "判定できませんでした"
            return claim
        verdict = str(data.get("verdict", "不明"))
        claim.verdict = verdict if verdict in VERDICTS else "不明"
        claim.note = strip_urls(str(data.get("note", "")))[:60]
        return claim


class FactCheckLoop(threading.Thread):
    """発話を溜めて、一定間隔で検出＋判定する（会議の字幕を止めないよう別スレッド）。"""

    def __init__(self, checker: FactChecker, on_claim: Callable[[Claim], None] | None = None) -> None:
        super().__init__(daemon=True)
        self.checker = checker
        self.on_claim = on_claim
        self._buffer: list[TranscriptSegment] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self.claims: list[Claim] = []

    def push(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self._buffer.append(segment)

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def _take(self) -> list[TranscriptSegment]:
        with self._lock:
            window, self._buffer = self._buffer, []
        return window

    def _process(self, window: list[TranscriptSegment]) -> None:
        for claim in self.checker.detect(window):
            self.checker.verify(claim)
            self.claims.append(claim)
            if self.on_claim is not None:
                self.on_claim(claim)

    def flush(self) -> list[Claim]:
        """残りを処理して、これまでに拾った主張を返す。"""
        self._process(self._take())
        return list(self.claims)

    def run(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(self.checker.cfg.interval_sec)
            self._wake.clear()
            if not self._stopping.is_set():
                self._process(self._take())


def strip_urls(text: str) -> str:
    """段1が混ぜてきた URL を落とす（モデルの記憶の URL は出典にならない）。"""
    import re

    cleaned = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()
