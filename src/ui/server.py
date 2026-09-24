"""ローカル会議 UI の FastAPI サーバーを扱う。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from src.ui.bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class UiConfig:
    """ローカル UI サーバーの設定を表す。"""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True


class UiActions(Protocol):
    """UI から会議本体へ依頼する操作を定義する。"""

    def rename_speaker(self, old: str, new: str) -> None:
        """話者名を変更する。"""

    def relabel_segment(self, start_time: float, end_time: float, old: str, new: str) -> None:
        """一発話だけ話者名を変更する。"""

    def session_info(self) -> dict:
        """セッション表示用の情報を返す。"""

    def participants(self) -> list[dict]:
        """参加者一覧を返す。"""

    def dispatch(self, todo_id: str) -> dict:
        """TODO の払い出しを実行して結果を返す。"""

    def verify_row(self, start_time: float, text: str) -> dict:
        """指された 1 行を Web で裏取りする（結果はイベントで流れてくる）。"""

    def review(self, reason: str) -> dict:
        """ここまでの全文を読み直して、決定事項・TODO を拾い直す。"""

    def set_planned_minutes(self, minutes: float | None) -> dict:
        """会議の予定時間（分）を受け取り、自動の見直し時刻を返す。"""

    def review_plan(self) -> dict:
        """予定時間と自動の見直し時刻を返す。"""

    def finish(self) -> dict:
        """会議を終わらせる。"""

    def answer_preflight(self, decision: str) -> dict:
        """起動セルフチェックの選択（start / retry / cancel）を受け取る。"""

    def answer_external_send(self, decision: str) -> dict:
        """録音を外へ出してよいかの回答（send / keep）を受け取る。"""

    def prep_status(self) -> dict:
        """いま読み込んでいる事前資料の一覧と、要約へ渡っている文字数を返す。"""

    def attach_prep(self, name: str, text: str) -> dict:
        """事前資料を足す（会議の途中でも）。"""

    def attach_prep_file(self, name: str, data: bytes) -> dict:
        """事前資料をファイルで足す（PDF・Word・Excel・PowerPoint は本文を取り出す）。"""

    def remove_prep(self, name: str) -> dict:
        """事前資料を外す。"""


def build_app(bus: EventBus, actions: UiActions, session_dir: Path, library_router=None) -> FastAPI:
    """指定したイベントバスと操作でローカル UI アプリを組み立てる。

    `library_router` を渡すと、画面「会議コックピット」（`/library`）も同じサーバーで開く（`src/ui/library.py`）。
    """
    app = FastAPI()
    if library_router is not None:
        app.include_router(library_router)
    index_path = Path(__file__).parent / "static" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """起動時ではなく要求時に HTML を読み込んで返す。"""
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/events")
    async def events(request: Request) -> StreamingResponse:
        """履歴と新着イベントを SSE として返す。"""
        subscription = bus.subscribe()

        async def stream() -> AsyncGenerator[str, None]:
            try:
                while not await request.is_disconnected():
                    event = await asyncio.to_thread(subscription.get, 15.0)
                    if event is None:
                        yield ": ping\n\n"
                    else:
                        yield f"data: {event.to_json()}\n\n"
            finally:
                subscription.close()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/state")
    def state() -> JSONResponse:
        """保存済みの会議状態を返す。"""
        state_path = session_dir / "state.json"
        if not state_path.exists():
            return JSONResponse({})
        return JSONResponse(json.loads(state_path.read_text(encoding="utf-8")))

    @app.get("/api/screens")
    def screens() -> JSONResponse:
        """いままでに控えた画面共有（会議中にも見られるように・2026-09-18 運用者 指摘）。

        「会議終了後にならないと見れないのね」と言われて足した。控えは撮ったそばから
        `screens.jsonl` に積んでいるので、会議中でもそのまま読める。
        """
        from src.screen.capture import pages as shared_pages

        found = shared_pages(session_dir)
        latest = ""
        path = session_dir / "screens.jsonl"
        if path.exists():
            for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    latest = str(json.loads(line).get("file", ""))
                except json.JSONDecodeError:
                    continue
                break
        return JSONResponse({"pages": found, "latest": latest, "shots": len(found)})

    @app.get("/screens/{name}")
    def screen_file(name: str) -> FileResponse:
        """控えた画像 1 枚。セッションの中だけを配る（名前に / や .. は通さない）。"""
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=404, detail="not found")
        path = session_dir / "screens" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.get("/api/status")
    def status() -> JSONResponse:
        """セッション情報と揮発的な最新イベントを返す。"""
        payload = dict(actions.session_info())
        for event_type in ("level", "metrics", "status"):
            event = bus.latest(event_type)
            if event is not None:
                payload[event_type] = event.data
        return JSONResponse(payload)

    @app.get("/api/participants")
    def participants() -> JSONResponse:
        """現在検出済みの参加者一覧を返す。"""
        return JSONResponse(actions.participants())

    @app.post("/api/rename")
    def rename(payload: dict) -> JSONResponse:
        """話者名の変更を会議本体へ依頼して UI に通知する。"""
        old = str(payload.get("old", "")).strip()
        new = str(payload.get("new", "")).strip()
        if not old or not new or old == new:
            raise HTTPException(status_code=400, detail="old and new must be different non-empty strings")
        if old == "不明話者?":
            raise HTTPException(status_code=400, detail="unknown speaker must be assigned per segment")
        try:
            actions.rename_speaker(old, new)
        except (KeyError, ValueError) as exc:
            # 未知の話者名・空名などは呼び出し側の誤りなので 400（500 にしない。3b のレビューで発見）
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        bus.publish("rename", {"old": old, "new": new})
        bus.publish("participants", {"items": actions.participants()})
        logger.info("Speaker renamed from %s to %s", old, new)
        return JSONResponse({"ok": True})

    @app.post("/api/relabel")
    def relabel(payload: dict) -> JSONResponse:
        """一発話だけの話者名変更を会議本体へ依頼して UI に通知する。"""
        old = str(payload.get("old", "")).strip()
        new = str(payload.get("new", "")).strip()
        start_time = payload.get("start_time")
        end_time = payload.get("end_time")
        if not old or not new or old == new:
            raise HTTPException(status_code=400, detail="old and new must be different non-empty strings")
        if isinstance(start_time, bool) or not isinstance(start_time, (int, float)):
            raise HTTPException(status_code=400, detail="start_time must be a number")
        if isinstance(end_time, bool) or not isinstance(end_time, (int, float)):
            raise HTTPException(status_code=400, detail="end_time must be a number")
        try:
            actions.relabel_segment(float(start_time), float(end_time), old, new)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        event = {"start_time": float(start_time), "end_time": float(end_time), "old": old, "new": new}
        bus.publish("relabel", event)
        bus.publish("participants", {"items": actions.participants()})
        logger.info("Segment relabeled from %s to %s at %.3f", old, new, start_time)
        return JSONResponse({"ok": True})

    # リプレイ画面の操作（ReplayUiActions）は見直しを持たない。無い操作は 500 ではなく
    #   「使えません」と返す（画面のボタンがエラーで固まらないように）。
    UNSUPPORTED = {"ok": False, "reason": "この画面では使えません"}

    @app.post("/api/review")
    def review(payload: dict) -> JSONResponse:
        """ここまでの全文の見直しを会議本体へ依頼する（画面のボタン）。"""
        run = getattr(actions, "review", None)
        if run is None:
            return JSONResponse(UNSUPPORTED)
        reason = str(payload.get("reason", "手動")).strip() or "手動"
        return JSONResponse(run(reason))

    @app.post("/api/external-send")
    def external_send(payload: dict) -> JSONResponse:
        """録音を外へ出してよいかの回答を受け取る。

        答えないまま閉じたときは送らない側に倒れる（待ち受け側が時間切れで keep にする）。
        """
        decision = str(payload.get("decision", "")).strip()
        try:
            return JSONResponse(actions.answer_external_send(decision))
        except ValueError as error:
            return JSONResponse({"ok": False, "error": str(error)}, status_code=400)

    @app.post("/api/preflight")
    def preflight(payload: dict) -> JSONResponse:
        """起動セルフチェックの選択（このまま開始／もう一度測る／中止）を会議本体へ渡す。"""
        answer = getattr(actions, "answer_preflight", None)
        if answer is None:
            return JSONResponse(UNSUPPORTED)
        decision = str(payload.get("decision", "")).strip()
        try:
            return JSONResponse(answer(decision))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/finish")
    def finish() -> JSONResponse:
        """会議を終わらせる（ターミナルで Ctrl+C を押さずに済むように）。"""
        stop = getattr(actions, "finish", None)
        if stop is None:
            return JSONResponse(UNSUPPORTED)
        result = stop()
        bus.publish("status", {"phase": "stopping", "warnings": [], "note": "終了処理に入りました。録音から全文を作り直します"})
        logger.info("Finish requested from the dashboard")
        return JSONResponse(result)

    @app.post("/api/finish-now")
    def finish_now(payload: dict) -> JSONResponse:
        """会議のあとの仕上げを「いま走らせる／あとでやる」。押されるまで重い処理は始まらない。"""
        choose = getattr(actions, "finish_now", None)
        if choose is None:
            return JSONResponse(UNSUPPORTED)
        choice = str(payload.get("choice", "")).strip().lower()
        if choice not in {"now", "later"}:
            return JSONResponse({"ok": False, "detail": "choice は now か later"}, status_code=400)
        logger.info("Finish choice from the dashboard: %s", choice)
        return JSONResponse(choose(choice))

    @app.post("/api/plan")
    def plan(payload: dict) -> JSONResponse:
        """会議の予定時間（分）を設定する。0 や空で自動の見直しを解除する。"""
        set_plan = getattr(actions, "set_planned_minutes", None)
        if set_plan is None:
            return JSONResponse(UNSUPPORTED)
        raw = payload.get("minutes")
        if raw in (None, "", 0):
            return JSONResponse(set_plan(None))
        try:
            minutes = float(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="minutes must be a number") from exc
        if not 1 <= minutes <= 600:
            raise HTTPException(status_code=400, detail="minutes must be between 1 and 600")
        result = set_plan(minutes)
        bus.publish("status", {"phase": "running", "warnings": [],
                               "note": f"予定 {minutes:g} 分（終了 {result.get('lead_min', 10):g} 分前に見直します）"})
        return JSONResponse(result)

    @app.get("/api/plan")
    def plan_status() -> JSONResponse:
        """予定時間と自動の見直し時刻を返す。"""
        plan_of = getattr(actions, "review_plan", None)
        return JSONResponse(plan_of() if plan_of else UNSUPPORTED)

    # ---------------------------------------------------- 事前資料（画面から添付）

    @app.get("/api/prep")
    def prep() -> JSONResponse:
        """いま読み込んでいる資料の一覧を返す。"""
        status = getattr(actions, "prep_status", None)
        return JSONResponse(status() if status else UNSUPPORTED)

    @app.post("/api/prep")
    def prep_add(payload: dict) -> JSONResponse:
        """資料を足す。中身は画面側でテキストとして読んで送る（PDF・Word は送れない）。"""
        attach = getattr(actions, "attach_prep", None)
        if attach is None:
            return JSONResponse(UNSUPPORTED)
        name = str(payload.get("name", "")).strip()
        text = payload.get("text", "")
        if not name or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="name and text are required")
        try:
            result = attach(name, text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"保存できません: {exc}") from exc
        logger.info("Prep material attached from the dashboard: %s", name)
        return JSONResponse(result)

    @app.post("/api/prep/file")
    def prep_add_file(payload: dict) -> JSONResponse:
        """ファイルを足す。中身は base64 で受け取り、本文の取り出しは**手元**で行う。

        PDF・Word・Excel・PowerPoint はブラウザでは文字にできないので、バイト列のまま受ける。
        """
        attach = getattr(actions, "attach_prep_file", None)
        if attach is None:
            return JSONResponse(UNSUPPORTED)
        name = str(payload.get("name", "")).strip()
        encoded = payload.get("base64", "")
        if not name or not isinstance(encoded, str) or not encoded:
            raise HTTPException(status_code=400, detail="name and base64 are required")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail="中身を読めませんでした") from exc
        try:
            result = attach(name, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"保存できません: {exc}") from exc
        logger.info("Prep file attached from the dashboard: %s (%d bytes)", name, len(data))
        return JSONResponse(result)

    @app.post("/api/prep/remove")
    def prep_remove(payload: dict) -> JSONResponse:
        """資料を外す。"""
        remove = getattr(actions, "remove_prep", None)
        if remove is None:
            return JSONResponse(UNSUPPORTED)
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        try:
            return JSONResponse(remove(name))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/client")
    def client() -> JSONResponse:
        """この会議のクライアント（いまの選択と、選べる一覧）。"""
        read = getattr(actions, "client_info", None)
        if read is None:
            raise HTTPException(status_code=404, detail="クライアントの選択は使えません")
        return JSONResponse(read())

    @app.post("/api/client")
    def choose_client(payload: dict) -> JSONResponse:
        """この会議のクライアントを決める。議事録・タスク・辞書の行き先が変わる。"""
        run = getattr(actions, "choose_client", None)
        if run is None:
            raise HTTPException(status_code=404, detail="クライアントの選択は使えません")
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="クライアント名が空です")
        try:
            entry = run(name, str(payload.get("client_id", "")), str(payload.get("how", "画面")))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        bus.publish("client", actions.client_info())
        return JSONResponse(entry)

    @app.post("/api/voice")
    def voice(payload: dict) -> JSONResponse:
        """声の台帳の候補を押した／断った。押したら改名として流す（過去の行にも反映される）。"""
        speaker = str(payload.get("speaker", "")).strip()
        name = str(payload.get("name", "")).strip()
        accepted = bool(payload.get("accepted"))
        run = getattr(actions, "accept_voice" if accepted else "reject_voice", None)
        if run is None:
            raise HTTPException(status_code=404, detail="声の台帳は使えません")
        if not speaker or not name:
            raise HTTPException(status_code=400, detail="speaker と name が要ります")
        try:
            result = run(speaker, name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if accepted:
            bus.publish("rename", {"old": speaker, "new": name})
        bus.publish("participants", {"items": actions.participants()})
        return JSONResponse(result)

    @app.post("/api/verify")
    def verify(payload: dict) -> JSONResponse:
        """画面で指された行の裏取りを頼む。結果は SSE の `verify` イベントで届く。"""
        run = getattr(actions, "verify_row", None)
        if run is None:
            return JSONResponse(UNSUPPORTED)
        start_time = payload.get("start_time")
        if isinstance(start_time, bool) or not isinstance(start_time, (int, float)):
            raise HTTPException(status_code=400, detail="start_time must be a number")
        try:
            return JSONResponse(run(float(start_time), str(payload.get("text", ""))))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dispatch")
    def dispatch(payload: dict) -> JSONResponse:
        """TODO の払い出しを会議本体へ依頼して UI に通知する。"""
        todo_id = str(payload.get("todo_id", "")).strip()
        if not todo_id:
            raise HTTPException(status_code=400, detail="todo_id must be a non-empty string")
        try:
            result = actions.dispatch(todo_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        bus.publish("dispatch", result)
        logger.info("TODO dispatched: %s", todo_id)
        return JSONResponse(result)

    return app


class PortInUse(OSError):
    """UI のポートが既に使われている（前のセッションが残っている）。"""


def ensure_port_free(host: str, port: int) -> None:
    """bind できなければ PortInUse を投げる。

    uvicorn は daemon スレッドの中で bind に失敗しても呼び出し側に伝えない。前のセッションが
    残っていると新しい起動は黙って失敗し、ブラウザは**古いサーバ**を見続ける（同じ URL なのに
    中身が別セッション。2026-09-09 運用者 の目視で発生、aa が lsof で特定）。起動前に自分で bind して確かめる。
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise PortInUse(
            f"ポート {host}:{port} は既に使われています。前のセッション（replay や会議モード）が残っています。"
            f"そのターミナルで Ctrl+C するか、`lsof -nP -iTCP:{port} -sTCP:LISTEN` で PID を確かめて止めてから起動してください。"
        ) from exc
    finally:
        probe.close()


PORT_TRIES = 10
"""既定のポートが塞がっていたら、いくつ隣まで試すか。"""


def start_ui_server(app: FastAPI, cfg: UiConfig) -> threading.Thread:
    """uvicorn を daemon スレッドで起動し、そのスレッドを返す。実際のポートは `thread.port`。

    塞がっていたら**隣のポートへ逃げる**（2026-09-17: 会議の 5 分前に、別のプロジェクトの
    プレビュー用サーバーが 8765 を掴んでいて会議を始められなかった）。会議は待ってくれないので、
    画面のポートごときで止めない。どうしても空きが無いときだけ PortInUse。
    """
    last: PortInUse | None = None
    for offset in range(PORT_TRIES):
        port = cfg.port + offset
        try:
            ensure_port_free(cfg.host, port)
        except PortInUse as exc:
            last = exc
            continue
        server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, name="meeting-ui", daemon=True)
        setattr(thread, "server", server)
        setattr(thread, "port", port)
        thread.start()
        if port != cfg.port:
            logger.warning("ポート %d が塞がっていたので %d で開きました", cfg.port, port)
        return thread
    raise last or PortInUse(f"{cfg.host}:{cfg.port} から {PORT_TRIES} 個ぶん、空きがありませんでした")
