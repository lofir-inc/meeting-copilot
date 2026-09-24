"""裏取り — 気になった発言を、頼まれたときだけ Web で確かめる。

自動では走らせない。2026-09-12 の実測では、自動検出は 50 分に 71 件拾い、その大半が
社内の見積・案件の話だった（外では確かめようがない）。人が「これは外の事実だ」と思った行だけを
調べるほうが、当たりも費用も良い。

調べるのは Claude CLI（サブスクの範囲・従量課金が増えない）。Gemini API は `google_search` を
有効にしないと**検索せず**、有効にすると 1,000 回 $14。Claude CLI は検索して**実際にページを開ける**
（記憶の URL は出典にならない — 実測で 29 本中 7 本が 404 だった）。

送るのは**その行と前後の文字起こしだけ**（音声は送らない）。

手段は `meeting.verify_engine`（auto | claude_cli | gemini | none）で選ぶ。auto は
①Claude CLI（サブスクがあれば）→ ②Gemini＋Google 検索（**その会議で外へのテキスト送信を承認したとき**だけ）
→ ③なし（🔍 を押すと理由を出す）。Claude のサブスクが無い人にも逃げ道を残す（2026-09-14 運用者 依頼）。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from src.llm.claude_cli_client import extract_json, validate_verification, verify_request

logger = logging.getLogger(__name__)

VERIFY_FILE = "verifications.jsonl"
"""画面から頼んだ裏取りの記録（発言・判定・出典）。議事録に出典を添えるための材料。"""

VERIFY_TIMEOUT_SEC = 120.0
"""裏取り 1 件の上限。これを過ぎたら「返事がありません」と出して終わる。

2026-09-14 の会議で、画面が「調べています…」のまま何も出なくなった。記録もログも残らず、
**始まったのかどうかも分からなかった**（会議の終了でスレッドごと消えた可能性が高い）。
始めたことを必ずログに残し、時間で必ず区切る。
"""

VERIFY_CONTEXT_SEC = 45.0
"""裏取りに添える前後の文字起こしの幅。発言だけだと、何の話か分からないことがある。"""

JST = ZoneInfo("Asia/Tokyo")


VERIFY_ENGINES = ("auto", "claude_cli", "gemini", "none")


SEARCH_THINKING_BUDGET = -1
"""裏取りのときだけ思考を「おまかせ」にする。実測（2026-09-15・gemini-3.8-flash）:
思考 0 と 1024 では `google_search` を付けても検索せず記憶で答えた（出典 0 件）。-1 だと検索することがある。
それでも検索するかはモデルが毎回決める（-1 で 7 回中 2 回）。聞き直しても検索しないことが多く、82 秒かかったのでやめた。
検索しなかった答えは「一致」にせず、画面に「検索されず、記憶で答えました」と出す。
"""


class GeminiVerifyClient:
    """Claude のサブスクが無い人向けの裏取り。Gemini に Google 検索を付けて調べる。

    作るのは**その会議で外へのテキスト送信が承認されたあと**だけ（呼び出し側の責任）。
    出典は、モデルが本文に書いた URL ではなく `groundingMetadata`＝**検索で実際に使われたページ**。
      モデルが書いた URL は記憶から出ることがあり、出典にならない。
    費用: 検索は月 5,000 回まで無料、超えると 1,000 回 $14。1 件ごとに人が押すので、まず届かない。
    """

    label = "Gemini＋Google 検索"

    def __init__(self, gemini) -> None:
        self.gemini = gemini

    def available(self) -> bool:
        return True

    def verify(self, quote: str, context: str = "", *, timeout_sec: int | None = None) -> dict:
        if not quote.strip():
            raise ValueError("発言が空です")
        text, grounded = self.gemini.search_json(
            verify_request(quote, context) + "\n必ず Google 検索で確かめてから答えること（記憶だけで答えない）。",
            thinking_budget=SEARCH_THINKING_BUDGET)
        data = extract_json(text)
        data["sources"] = [source["uri"] for source in grounded]
        result = validate_verification(data)
        if not result["sources"]:
            # 検索しなかった＝記憶で答えた。「一致」をそのまま出すと、確かめたように見える
            if result["verdict"] == "一致":
                result["verdict"] = "要確認"
            result["note"] = f"（検索されず、記憶で答えました）{result['note']}"[:200]
        return result


class Verifier:
    """画面から頼まれた 1 行を、裏で調べて結果を流す。"""

    def __init__(
        self,
        session_dir: Path,
        client,
        *,
        quote_at: Callable[[float], str],
        context_at: Callable[[float], str],
        publish: Callable[[str, dict], None],
        context_sec: float = VERIFY_CONTEXT_SEC,
        timeout_sec: float = VERIFY_TIMEOUT_SEC,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.client = client
        """裏取りの手段。None なら使えない（`unavailable_reason` を画面に出す）。"""
        self.label = getattr(client, "label", "Claude CLI") if client is not None else ""
        self.unavailable_reason = ""
        self._quote_at = quote_at
        self._context_at = context_at
        self._publish = publish
        self.context_sec = context_sec
        self.timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._running: set[float] = set()
        """いま調べている行（同じ行を二重に投げない）。"""

    def request(self, start_time: float, text: str = "") -> dict:
        """画面で指された 1 行の裏取りを頼む（**裏で走る**。結果は画面へ流す）。"""
        quote = (text or "").strip() or self._quote_at(start_time)
        if not quote:
            raise KeyError(f"その行が見つかりません: {start_time}")
        if self.client is None or not self.client.available():
            return {"ok": False, "reason": f"裏取りは使えません（{self.unavailable_reason or 'Claude CLI が見つかりません'}）"}
        key = round(float(start_time), 2)
        with self._lock:
            if key in self._running:
                return {"ok": True, "start_time": key, "phase": "running", "already": True}
            self._running.add(key)
        payload = {"start_time": key, "quote": quote, "phase": "running",
                   "timeout_sec": self.timeout_sec, "engine": self.label}
        logger.info("裏取りを始めます（%.1f 秒の行）: %s", key, quote[:40])
        self._publish("verify", payload)
        threading.Thread(target=self._work, args=(key, quote), daemon=True, name="verify").start()
        return {"ok": True, **payload}

    def _work(self, start_time: float, quote: str) -> None:
        """裏取りを 1 件走らせて、結果を画面と記録へ流す（会議は止めない）。"""
        started = time.time()
        try:
            result = self.client.verify(quote, self._context_at(start_time),
                                        timeout_sec=int(self.timeout_sec))
            payload = {"start_time": start_time, "quote": quote, "phase": "done", **result,
                       "seconds": round(time.time() - started, 1)}
        except Exception as error:  # noqa: BLE001 — 調べられなくても会議は続く
            logger.warning("裏取りに失敗しました（%.1f 秒の行）: %s", start_time, error)
            payload = {"start_time": start_time, "quote": quote, "phase": "done",
                       "verdict": "不明", "note": f"調べられませんでした（{str(error)[:80]}）",
                       "sources": [], "seconds": round(time.time() - started, 1)}
        finally:
            with self._lock:
                self._running.discard(start_time)
        self._record(payload)
        self._publish("verify", payload)
        logger.info("裏取り %s: %s（%s 秒・出典 %d 件）", payload["verdict"], quote[:40],
                    payload["seconds"], len(payload.get("sources", [])))

    def use(self, client, label: str = "") -> None:
        """手段を差し替える（会議の承認が決まったあとに Gemini へ、など）。"""
        self.client = client
        self.label = label or getattr(client, "label", "Claude CLI")
        self.unavailable_reason = ""

    def disable(self, reason: str) -> None:
        """使える手段が無い。🔍 を押したときに、なぜ使えないかを出す。"""
        self.client = None
        self.label = ""
        self.unavailable_reason = reason

    def model_info(self) -> dict | None:
        """画面の「モデル」欄に出す形。"""
        if self.client is None:
            return {"name": "なし", "where": self.unavailable_reason or "使えません", "available": False}
        where = "外（Gemini・テキストのみ）" if isinstance(self.client, GeminiVerifyClient) else "外（Claude CLI・テキストのみ）"
        return {"name": self.label, "where": where, "available": True}

    def unfinished(self) -> list[float]:
        """いま調べている途中の行（会議の終わりに「黙って消える」のを知らせるため）。"""
        with self._lock:
            return sorted(self._running)

    def _record(self, payload: dict) -> None:
        try:
            with (self.session_dir / VERIFY_FILE).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": datetime.now(JST).isoformat(timespec="seconds"),
                                         **payload}, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("裏取りの記録に失敗")
