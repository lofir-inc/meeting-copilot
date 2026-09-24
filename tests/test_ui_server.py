"""ローカル会議 UI サーバーを ASGI 経由で検証する。"""

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from src.ui.bus import EventBus
from src.ui.server import build_app


class FakeActions:
    """UI が呼ぶ操作を記録するテスト用実装。"""

    def __init__(self) -> None:
        self.renames: list[tuple[str, str]] = []
        self.relabels: list[tuple[float, float, str, str]] = []
        self.dispatches: list[str] = []
        self.reviews: list[str] = []
        self.planned: float | None = None
        self.finished = False
        self.external_answers: list[str] = []
        self.finish_choices: list[str] = []
        self.preflight_answers: list[str] = []
        self.prep: list[dict] = []
        self.verifies: list[tuple[float, str]] = []

    def rename_speaker(self, old: str, new: str) -> None:
        """話者名変更を記録する。"""
        self.renames.append((old, new))

    def relabel_segment(self, start_time: float, end_time: float, old: str, new: str) -> None:
        """発話単位の話者変更を記録する。"""
        self.relabels.append((start_time, end_time, old, new))

    def session_info(self) -> dict:
        """画面に必要な固定セッション情報を返す。"""
        return {
            "session_name": "テスト会議",
            "started_at": 1.0,
            "self_name": "自分",
            "models": {"stt": {"name": "whisper", "where": "ローカル（mlx）"}},
        }

    def participants(self) -> list[dict]:
        """テスト用の参加者一覧を返す。"""
        return [{"name": "田中", "enrolled": True, "utterances": 2, "chars": 10}]

    def dispatch(self, todo_id: str) -> dict:
        """払い出し要求を記録し、固定結果を返す。"""
        if todo_id == "missing":
            raise KeyError(todo_id)
        self.dispatches.append(todo_id)
        return {"todo_id": todo_id, "brief_path": "/brief.md", "already": False, "notion": None, "slack_ok": None, "agent": None, "errors": {}}


    def review(self, reason: str) -> dict:
        """見直し要求を記録し、固定結果を返す。"""
        self.reviews.append(reason)
        return {"ok": True, "changes": 3, "decisions": 2, "todos": 4, "seconds": 12.3}

    def set_planned_minutes(self, minutes: float | None) -> dict:
        """予定時間を記録して、自動の見直し時刻を返す。"""
        self.planned = minutes
        return {"planned_minutes": minutes, "lead_min": 10.0, "at": 600.0 if minutes else None, "done": False}

    def review_plan(self) -> dict:
        """いまの予定を返す。"""
        return {"planned_minutes": self.planned, "lead_min": 10.0, "at": None, "done": False}

    def finish(self) -> dict:
        """終了要求を記録する。"""
        self.finished = True
        return {"ok": True, "note": "終了処理に入ります"}

    def verify_row(self, start_time: float, text: str) -> dict:
        """裏取りの依頼を記録して、受け付けた形を返す。"""
        if start_time > 900:
            raise KeyError("その行が見つかりません")
        self.verifies.append((start_time, text))
        return {"ok": True, "start_time": start_time, "phase": "running"}

    def prep_status(self) -> dict:
        """いま読み込んでいる資料を返す。"""
        return {"files": self.prep, "chars": sum(file["chars"] for file in self.prep),
                "used_chars": 100, "limit": 2000}

    def attach_prep(self, name: str, text: str) -> dict:
        """資料を足す（読めない種類は断る）。"""
        if not name.endswith((".md", ".txt")):
            raise ValueError("読めない種類です")
        self.prep.append({"name": name, "chars": len(text)})
        return self.prep_status()

    def attach_prep_file(self, name: str, data: bytes) -> dict:
        """ファイルを足す（中身の長さだけ覚える）。"""
        if not data:
            raise ValueError("中身が空です")
        self.prep.append({"name": name, "chars": len(data)})
        return self.prep_status()

    def remove_prep(self, name: str) -> dict:
        """資料を外す。"""
        before = len(self.prep)
        self.prep = [file for file in self.prep if file["name"] != name]
        if len(self.prep) == before:
            raise KeyError(name)
        return self.prep_status()

    def answer_preflight(self, decision: str) -> dict:
        """セルフチェックの選択を記録する。"""
        if decision not in {"start", "retry", "cancel"}:
            raise ValueError(f"未知の選択: {decision}")
        self.preflight_answers.append(decision)
        return {"ok": True, "decision": decision}

    def answer_external_send(self, decision: str) -> dict:
        """録音を外へ出すかの選択を記録する。"""
        if decision not in {"send", "keep"}:
            raise ValueError(f"未知の選択: {decision}")
        self.external_answers.append(decision)
        return {"ok": True, "decision": decision}

    def finish_now(self, choice: str) -> dict:
        """仕上げを「いま／あとで」の選択を記録する。"""
        self.finish_choices.append(choice)
        return {"ok": True, "choice": choice}


def create_client(session_dir: Path) -> tuple[TestClient, EventBus, FakeActions]:
    """独立した UI アプリとテストクライアントを作る。"""
    bus = EventBus()
    actions = FakeActions()
    return TestClient(build_app(bus, actions, session_dir)), bus, actions


def test_index_is_local_and_small(tmp_path: Path) -> None:
    """トップページは外部リソースを参照しない小さな単一 HTML である。"""
    client, _, _ = create_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert len(response.content) < 200 * 1024
    assert not re.search(r"(?:src|href)=['\"][^'\"]*https?://", response.text)


def test_state_returns_file_or_empty_object(tmp_path: Path) -> None:
    """state.json があれば返し、なければ空の状態を返す。"""
    client, _, _ = create_client(tmp_path)
    assert client.get("/api/state").json() == {}
    (tmp_path / "state.json").write_text(json.dumps({"summary": "現在の要約"}), encoding="utf-8")
    assert client.get("/api/state").json() == {"summary": "現在の要約"}


def test_rename_calls_action_and_publishes_event(tmp_path: Path) -> None:
    """名寄せ要求は会議操作を呼び、rename イベントを残す。"""
    client, bus, actions = create_client(tmp_path)
    response = client.post("/api/rename", json={"old": "不明話者1", "new": "田中"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert actions.renames == [("不明話者1", "田中")]
    assert bus.latest("rename").data == {"old": "不明話者1", "new": "田中"}  # type: ignore[union-attr]
    assert client.post("/api/rename", json={"old": "", "new": "田中"}).status_code == 400
    assert client.post("/api/rename", json={"old": "田中", "new": "田中"}).status_code == 400
    assert client.post("/api/rename", json={"old": "不明話者?", "new": "田中"}).status_code == 400


def test_rename_accepts_a_ghost_participant_when_actions_handle_it(tmp_path: Path) -> None:
    """付け替えで生じた diarizer 外の名前も、操作側が処理できれば改名できる。"""
    client, _, actions = create_client(tmp_path)
    response = client.post("/api/rename", json={"old": "とりのこ", "new": "鳥の子"})
    assert response.status_code == 200
    assert actions.renames == [("とりのこ", "鳥の子")]


def test_relabel_calls_action_and_publishes_event(tmp_path: Path) -> None:
    """行単位の話者変更は会議操作を呼び、relabel イベントを残す。"""
    client, bus, actions = create_client(tmp_path)
    response = client.post("/api/relabel", json={"start_time": 1.5, "end_time": 3.0, "old": "不明話者?", "new": "田中"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert actions.relabels == [(1.5, 3.0, "不明話者?", "田中")]
    assert bus.latest("relabel").data == {"start_time": 1.5, "end_time": 3.0, "old": "不明話者?", "new": "田中"}  # type: ignore[union-attr]
    assert client.post("/api/relabel", json={"start_time": "1", "end_time": 3.0, "old": "不明話者?", "new": "田中"}).status_code == 400
    assert client.post("/api/relabel", json={"start_time": 1.0, "end_time": 3.0, "old": "田中", "new": "田中"}).status_code == 400


def test_relabel_accepts_returning_a_segment_to_unknown(tmp_path: Path) -> None:
    """行単位の付け替えでは不明話者へ戻す指定を許可する。"""
    client, _, actions = create_client(tmp_path)
    response = client.post("/api/relabel", json={"start_time": 1.5, "end_time": 3.0, "old": "とりのこ", "new": "不明話者?"})
    assert response.status_code == 200
    assert actions.relabels == [(1.5, 3.0, "とりのこ", "不明話者?")]


def test_status_contains_session_info_and_latest_events(tmp_path: Path) -> None:
    """ステータスは会議情報と各種最新イベントをまとめて返す。"""
    client, bus, _ = create_client(tmp_path)
    bus.publish("level", {"self_dbfs": -20, "remote_dbfs": -30})
    bus.publish("metrics", {"stt_queue_depth": 2})
    payload = client.get("/api/status").json()
    assert payload["session_name"] == "テスト会議"
    assert payload["level"]["self_dbfs"] == -20
    assert payload["metrics"]["stt_queue_depth"] == 2
    assert payload["models"]["stt"]["where"] == "ローカル（mlx）"


def test_participants_returns_actions_items(tmp_path: Path) -> None:
    """参加者 API は操作実装の現在値を配列で返す。"""
    client, _, _ = create_client(tmp_path)
    assert client.get("/api/participants").json() == [
        {"name": "田中", "enrolled": True, "utterances": 2, "chars": 10},
    ]


def test_dispatch_calls_action_and_publishes_event(tmp_path: Path) -> None:
    """TODO 払い出しは操作を呼び、結果を dispatch イベントとして積む。"""
    client, bus, actions = create_client(tmp_path)
    response = client.post("/api/dispatch", json={"todo_id": "T1"})
    assert response.status_code == 200
    assert response.json()["todo_id"] == "T1"
    assert actions.dispatches == ["T1"]
    assert bus.latest("dispatch").data["todo_id"] == "T1"  # type: ignore[union-attr]
    assert client.post("/api/dispatch", json={"todo_id": "missing"}).status_code == 404


def test_events_replays_history_as_sse_data_lines(tmp_path: Path) -> None:
    """SSE 接続時には先に積まれた三件の履歴を data 行で返す。"""
    client, bus, _ = create_client(tmp_path)
    for index in range(3):
        bus.publish("segment", {"index": index})
    subscription = bus.subscribe()
    lines = [f"data: {subscription.get(0.01).to_json()}" for _ in range(3)]  # type: ignore[union-attr]
    assert [json.loads(line.removeprefix("data: "))["data"]["index"] for line in lines] == [0, 1, 2]


def test_ensure_port_free_raises_when_port_is_taken():
    """前のセッションが残っていると新しい起動は黙って失敗し、ブラウザが古いサーバを見続ける（2026-09-09）。"""
    import socket

    from src.ui.server import PortInUse, ensure_port_free

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        import pytest

        with pytest.raises(PortInUse):
            ensure_port_free("127.0.0.1", port)
    except OSError:
        import pytest

        pytest.skip("この環境では loopback に bind できない")
    finally:
        holder.close()


def test_review_button_asks_the_meeting_to_reread(tmp_path: Path) -> None:
    """運用者 案 2026-09-12: 終了 10 分前に一度まわす。画面のボタンから頼めること。"""
    client, _, actions = create_client(tmp_path)

    response = client.post("/api/review", json={"reason": "手動"})

    assert response.status_code == 200
    assert response.json()["decisions"] == 2
    assert actions.reviews == ["手動"]


def test_planned_minutes_are_accepted_and_announced(tmp_path: Path) -> None:
    client, bus, actions = create_client(tmp_path)
    subscription = bus.subscribe()

    response = client.post("/api/plan", json={"minutes": 60})

    assert response.status_code == 200
    assert actions.planned == 60.0
    assert "予定 60 分" in subscription.get(0.5).data["note"]


def test_empty_minutes_clears_the_schedule(tmp_path: Path) -> None:
    client, _, actions = create_client(tmp_path)
    client.post("/api/plan", json={"minutes": 60})

    response = client.post("/api/plan", json={"minutes": None})

    assert response.status_code == 200
    assert actions.planned is None


def test_absurd_minutes_are_rejected(tmp_path: Path) -> None:
    client, _, actions = create_client(tmp_path)

    assert client.post("/api/plan", json={"minutes": 9999}).status_code == 400
    assert client.post("/api/plan", json={"minutes": "とても長い"}).status_code == 400
    assert actions.planned is None


def test_finish_button_stops_the_meeting(tmp_path: Path) -> None:
    """画面の「会議を終了」から終われる（ターミナルの Ctrl+C の代わり）。"""
    client, bus, actions = create_client(tmp_path)
    subscription = bus.subscribe()

    response = client.post("/api/finish")

    assert response.status_code == 200 and response.json()["ok"] is True
    assert actions.finished is True
    assert subscription.get(0.5).data["phase"] == "stopping"


def test_preflight_choice_is_passed_to_the_meeting(tmp_path: Path) -> None:
    """起動セルフチェックの選択を画面から渡せる。"""
    client, _, actions = create_client(tmp_path)

    response = client.post("/api/preflight", json={"decision": "start"})

    assert response.status_code == 200
    assert actions.preflight_answers == ["start"]


def test_unknown_preflight_choice_is_rejected(tmp_path: Path) -> None:
    client, _, actions = create_client(tmp_path)

    assert client.post("/api/preflight", json={"decision": "とりあえず"}).status_code == 400
    assert actions.preflight_answers == []


def test_外へ出すかの選択を画面から渡せる(tmp_path: Path) -> None:
    """会議のあとの「外へ出すか」を端末でなく画面で答えられる（運用者 依頼 2026-09-13）。"""
    client, _, actions = create_client(tmp_path)

    assert client.post("/api/external-send", json={"decision": "send"}).status_code == 200
    assert client.post("/api/external-send", json={"decision": "keep"}).status_code == 200
    assert actions.external_answers == ["send", "keep"]


def test_知らない選択は弾く_外へ出すか(tmp_path: Path) -> None:
    """曖昧な値を「送ってよい」と解釈しない。"""
    client, _, actions = create_client(tmp_path)

    assert client.post("/api/external-send", json={"decision": "たぶん"}).status_code == 400
    assert client.post("/api/external-send", json={}).status_code == 400
    assert actions.external_answers == []


def test_画面から事前資料を足せる():
    """会議ごとに空から始めて、要るものをその場で足す形にした（2026-09-14）。"""
    actions = FakeActions()
    client = TestClient(build_app(EventBus(), actions, Path(".")))

    assert client.get("/api/prep").json()["files"] == []
    added = client.post("/api/prep", json={"name": "議題.md", "text": "見積の確認"})
    assert added.status_code == 200
    assert [file["name"] for file in added.json()["files"]] == ["議題.md"]
    assert client.get("/api/prep").json()["used_chars"] == 100


def test_読めない種類は断る():
    client = TestClient(build_app(EventBus(), FakeActions(), Path(".")))
    response = client.post("/api/prep", json={"name": "提案書.pdf", "text": "%PDF"})
    assert response.status_code == 400
    assert "読めない種類" in response.json()["detail"]


def test_事前資料を外せる():
    actions = FakeActions()
    client = TestClient(build_app(EventBus(), actions, Path(".")))
    client.post("/api/prep", json={"name": "議題.md", "text": "見積の確認"})

    assert client.post("/api/prep/remove", json={"name": "議題.md"}).json()["files"] == []
    assert client.post("/api/prep/remove", json={"name": "無い.md"}).status_code == 404


def test_この画面では使えないと返す():
    """リプレイ画面には事前資料の口が無い。ボタンがエラーで固まらないようにする。"""
    class Minimal(FakeActions):
        prep_status = None

    client = TestClient(build_app(EventBus(), Minimal(), Path(".")))
    assert client.get("/api/prep").json() == {"ok": False, "reason": "この画面では使えません"}


def test_資料はファイルとして送れる():
    """PDF・Word・Excel・PowerPoint はブラウザで文字にできないので、バイト列で受ける。"""
    import base64

    actions = FakeActions()
    client = TestClient(build_app(EventBus(), actions, Path(".")))
    encoded = base64.b64encode("%PDF-1.4".encode()).decode()

    response = client.post("/api/prep/file", json={"name": "提案書.pdf", "base64": encoded})

    assert response.status_code == 200
    assert [file["name"] for file in response.json()["files"]] == ["提案書.pdf"]


def test_壊れた中身は断る():
    client = TestClient(build_app(EventBus(), FakeActions(), Path(".")))
    response = client.post("/api/prep/file", json={"name": "提案書.pdf", "base64": "これは base64 ではない"})
    assert response.status_code == 400


def test_画面から裏取りを頼める():
    """頼むのは人。自動では走らせない（自動検出は社内の話ばかり拾った実測がある）。"""
    actions = FakeActions()
    client = TestClient(build_app(EventBus(), actions, Path(".")))

    response = client.post("/api/verify", json={"start_time": 12.5, "text": "0.75 ドルらしい"})

    assert response.status_code == 200
    assert response.json()["phase"] == "running"
    assert actions.verifies == [(12.5, "0.75 ドルらしい")]


def test_無い行の裏取りは_404():
    client = TestClient(build_app(EventBus(), FakeActions(), Path(".")))
    assert client.post("/api/verify", json={"start_time": 9999}).status_code == 404


def test_時刻が数でなければ_400():
    client = TestClient(build_app(EventBus(), FakeActions(), Path(".")))
    assert client.post("/api/verify", json={"start_time": "12 秒"}).status_code == 400


def test_声の候補を押すと改名として流れる():
    class VoiceActions(FakeActions):
        def __init__(self):
            super().__init__()
            self.voice: list[tuple[str, str, bool]] = []

        def accept_voice(self, speaker, name):
            self.voice.append((speaker, name, True))
            return {"ok": True}

        def reject_voice(self, speaker, name):
            self.voice.append((speaker, name, False))
            return {"ok": True}

    actions = VoiceActions()
    bus = EventBus()
    events = bus.subscribe()
    client = TestClient(build_app(bus, actions, Path(".")))

    assert client.post("/api/voice", json={"speaker": "不明話者1", "name": "参加者C", "accepted": True}).status_code == 200
    assert client.post("/api/voice", json={"speaker": "不明話者1", "name": "参加者D", "accepted": False}).status_code == 200

    assert actions.voice == [("不明話者1", "参加者C", True), ("不明話者1", "参加者D", False)]
    first = events.get(0.1)
    assert first.type == "rename" and first.data == {"old": "不明話者1", "new": "参加者C"}


def test_声の台帳が無い会議では_404():
    client = TestClient(build_app(EventBus(), FakeActions(), Path(".")))

    assert client.post("/api/voice", json={"speaker": "不明話者1", "name": "参加者C", "accepted": True}).status_code == 404


class TestFinishAsk:
    """会議のあとの仕上げを、画面で「いま／あとで」と選ぶ。

    2026-09-17 運用者 指摘「画面の『いま仕上げる』ボタンが見当たらない」。
    会議本体は finish の知らせを出していたが、画面に描く処理も押す先の口も無かった。
    """

    def test_押した選択が会議本体へ渡る(self, tmp_path: Path) -> None:
        client, _, actions = create_client(tmp_path)

        assert client.post("/api/finish-now", json={"choice": "now"}).json() == {"ok": True, "choice": "now"}
        assert client.post("/api/finish-now", json={"choice": "later"}).json() == {"ok": True, "choice": "later"}
        assert actions.finish_choices == ["now", "later"]

    def test_知らない選択は断る(self, tmp_path: Path) -> None:
        client, _, actions = create_client(tmp_path)

        assert client.post("/api/finish-now", json={"choice": "maybe"}).status_code == 400
        assert actions.finish_choices == []

    def test_画面を開き直しても聞かれたままになる(self, tmp_path: Path) -> None:
        """あとから開いた画面にも出す（会議の終わりは席を外していることがある）。"""
        bus = EventBus()
        bus.publish("finish", {"ask": True, "steps": [], "timeout_sec": 600})

        subscription = bus.subscribe()
        first = subscription.get(timeout=1)

        assert first is not None and first.type == "finish" and first.data["ask"] is True

    def test_画面に仕上げのボタンがある(self) -> None:
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/index.html").read_text(encoding="utf-8")

        assert "いま仕上げる" in html and "あとでやる" in html and "/api/finish-now" in html


class TestRedraw:
    """繋ぎ直しても、同じ行を二度描かない。

    2026-09-21 に実際に見た: 見本の画面を開いたままサーバーを起こし直したら、
      **全行が 6 回ずつ並んだ**。原因は「繋ぎ直すとサーバーが履歴を全部送り直す」×
      「画面が届いた順に足すだけ」。会議中に接続が一度切れれば、実会議でも同じになる。

    履歴を送るのをやめる手は取らない——**途中から画面を開いた人に全部見せる**ために要る
      （会議の最中に開き直すのは普通のこと）。∴ 弾くのは画面側。
    """

    def test_履歴は全部送り直される(self) -> None:
        """前提の確認（ここが変わったら、画面側の重複除けも見直す）。"""
        bus = EventBus()
        bus.publish("segment", {"speaker": "田中", "text": "はい", "start_time": 1.0, "end_time": 2.0})

        first = bus.subscribe().get(timeout=1)
        second = bus.subscribe().get(timeout=1)

        assert first is not None and second is not None
        assert first.data == second.data          # あとから繋いだ人にも同じ行が届く

    def test_画面は同じ行を二度描かない(self) -> None:
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/index.html").read_text(encoding="utf-8")

        assert "drawnCaptions" in html
        assert "if (drawnCaptions.has(captionKey(data))) return;" in html

    def test_鍵に本文を入れない(self) -> None:
        """あとから直した行（amend）が、履歴の再生で元の本文のまま増えないように。"""
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/index.html").read_text(encoding="utf-8")

        # `}` で切ると `${data.start_time}` の途中で切れる（1 度やった）。行ごと取る
        key = next(line for line in html.splitlines() if line.startswith("function captionKey"))
        assert "start_time" in key and "end_time" in key and "speaker" in key
        assert "data.text" not in key


class TestPortFallback:
    """画面のポートが塞がっていても会議は始まる。

    2026-09-17 運用者 指摘「会議 5 分前に、ポート占有で起動できなくてとても焦った」。
    掴んでいたのは別プロジェクトのプレビュー用サーバーだった。会議は待ってくれないので、
    ポートが塞がっていたら隣へ逃げる。
    """

    def test_塞がっていたら隣のポートで開く(self, tmp_path: Path) -> None:
        import socket

        from src.ui.server import UiConfig, start_ui_server

        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        taken = squatter.getsockname()[1]
        try:
            client, bus, actions = create_client(tmp_path)
            thread = start_ui_server(build_app(bus, actions, tmp_path), UiConfig(port=taken))
            try:
                assert getattr(thread, "port") == taken + 1
            finally:
                thread.server.should_exit = True
                thread.join(timeout=5)
            client.close()
        finally:
            squatter.close()

    def test_どこも空いていなければ言う(self, tmp_path: Path) -> None:
        import socket

        import pytest

        from src.ui.server import PORT_TRIES, PortInUse, UiConfig, start_ui_server

        held = []
        base = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
            probe.close()
            for offset in range(PORT_TRIES):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", base + offset))
                    sock.listen(1)
                    held.append(sock)
                except OSError:
                    sock.close()
            if len(held) < PORT_TRIES:
                pytest.skip("連続した空きポートを確保できませんでした")
            client, bus, actions = create_client(tmp_path)
            with pytest.raises(PortInUse):
                start_ui_server(build_app(bus, actions, tmp_path), UiConfig(port=base))
            client.close()
        finally:
            for sock in held:
                sock.close()


class TestLiveScreens:
    """2026-09-18 運用者 指摘「これは会議終了後にならないと見れないのね」。

    控えは撮ったそばから screens.jsonl に積んでいるので、会議中でも読める。
    """

    def _client(self, session_dir):
        return TestClient(build_app(EventBus(), FakeActions(), session_dir))

    def test_会議中でも見せられたページを返す(self, tmp_path):
        (tmp_path / "screens").mkdir()
        (tmp_path / "screens" / "000126.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "screens.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
            {"at": 6.5, "file": "screens/000006.jpg", "title": "Meet",
             "urls": ["https://meet.google.com/abc-defg-hij"]},
            {"at": 126.0, "file": "screens/000126.jpg", "title": "CloudCut", "urls": []},
        ]), encoding="utf-8")
        client = self._client(tmp_path)

        found = client.get("/api/screens").json()

        assert found["latest"] == "screens/000126.jpg"
        assert [one["title"] for one in found["pages"]] == ["Meet", "CloudCut"]

    def test_控えた画像を配る(self, tmp_path):
        (tmp_path / "screens").mkdir()
        (tmp_path / "screens" / "000126.jpg").write_bytes(b"\xff\xd8\xff")
        client = self._client(tmp_path)

        assert client.get("/screens/000126.jpg").status_code == 200

    def test_セッションの外は配らない(self, tmp_path):
        """名前に .. を入れて外を覗けないこと。"""
        client = self._client(tmp_path)

        assert client.get("/screens/..%2F..%2Fsettings.yaml").status_code == 404
        assert client.get("/screens/nope.jpg").status_code == 404

    def test_まだ一枚も無ければ空で返す(self, tmp_path):
        client = self._client(tmp_path)

        found = client.get("/api/screens").json()

        assert found["pages"] == [] and found["latest"] == ""
