"""画面「辞書と声の台帳」— 辞書の候補を選ぶ・辞書を直す・声の名前を直す（ファイルを手で編集しない）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.text import glossary_edit
from src.text.glossary_candidates import CANDIDATES_JSON
from src.ui.library import Library, build_library_router

GLOSSARY = """# Gemini 用の辞書（コメントは消さない）
replacements:
  クラウド: Claude   # 守り札が要る
  自社: 自社

protect:
  - クラウドファンディング
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "glossary.yaml").write_text("replacements:\n  ギットハブ: GitHub\n", encoding="utf-8")
    (tmp_path / "config" / "glossary-gemini.yaml").write_text(GLOSSARY, encoding="utf-8")
    session = tmp_path / "workspace" / "sessions" / "2026-09-14_1359"
    session.mkdir(parents=True)
    (session / "external_sends.jsonl").write_text(json.dumps({"note": "live_batch"}) + "\n", encoding="utf-8")
    (session / CANDIDATES_JSON).write_text(json.dumps([
        {"kind": "replace", "wrong": "レブノル", "right": "ITレビュー", "count": 2, "source": "議事録の LLM", "reason": "", "examples": []},
        {"kind": "protect", "wrong": "MDクラウド", "right": "", "count": 1, "source": "辞書が壊した跡", "reason": "", "examples": []},
        {"kind": "replace", "wrong": "経団連", "right": "経営層", "count": 1, "source": "議事録の LLM", "reason": "", "examples": []},
    ], ensure_ascii=False), encoding="utf-8")
    return tmp_path


def settings():
    return {"meeting": {"glossary_path": "config/glossary.yaml",
                        "external_stt": {"glossary_path": "config/glossary-gemini.yaml"},
                        "voice_library": {"enabled": True, "path": "workspace/voices/library.json"}}}


@pytest.fixture
def library(repo):
    changes, pushed = [], []
    lib = Library(repo, settings, on_change=changes.append,
                  push_to_task_hub=lambda pairs, session: pushed.extend(pairs) or [{"status": "added"}])
    lib.changes, lib.pushed = changes, pushed
    return lib


class TestDictionary:
    def test_辞書を一覧で見る(self, library):
        gemini = next(d for d in library.dictionaries() if d["id"] == "gemini")

        assert gemini["replacements"][0] == {"wrong": "クラウド", "right": "Claude"}
        assert gemini["protect"] == ["クラウドファンディング"]

    def test_足すとタスク管理にも候補で送り会議中なら読み直す(self, library, repo):
        result = library.add("gemini", "レブノル", "Revuno")

        assert result["added"] is True
        assert yaml.safe_load((repo / "config/glossary-gemini.yaml").read_text())["replacements"]["レブノル"] == "Revuno"
        assert library.pushed == [("レブノル", "Revuno")] and library.changes == ["glossary"]

    def test_外すときは退避してからコメントを残して外す(self, library, repo):
        result = library.remove("gemini", "自社")

        text = (repo / "config/glossary-gemini.yaml").read_text(encoding="utf-8")
        assert "自社" not in text and "# 守り札が要る" in text and "# Gemini 用の辞書" in text
        assert "自社" in Path(result["backup"]).read_text(encoding="utf-8")

    def test_守り札を足して外す(self, library, repo):
        library.add_guard("gemini", "MDクラウド")
        library.remove_guard("gemini", "クラウドファンディング")

        assert yaml.safe_load((repo / "config/glossary-gemini.yaml").read_text())["protect"] == ["MDクラウド"]

    def test_守り札を誤りとして足すのは止める(self, library):
        with pytest.raises(ValueError):
            library.add("gemini", "クラウドファンディング", "x")

    def test_いま使っている辞書に足す(self, repo):
        lib = Library(repo, settings, current_dictionary=lambda: "gemini")

        lib.add("current", "ジェンマ", "Gemma")

        assert "ジェンマ" in (repo / "config/glossary-gemini.yaml").read_text(encoding="utf-8")


class TestCandidates:
    def test_会議を選ぶと入れる先の辞書も分かる(self, library):
        assert library.sessions() == [{"name": "2026-09-14_1359", "pending": 3}]
        assert library.candidates("2026-09-14_1359")["dictionary"] == "gemini"    # 外で起こした会議

    def test_入れる見送るを決めると残りから消える(self, library, repo):
        result = library.decide("2026-09-14_1359", [
            {"kind": "replace", "wrong": "レブノル", "right": "Revuno", "accept": True},   # 画面で直した表記で入る
            {"kind": "protect", "wrong": "MDクラウド", "accept": True},
            {"kind": "replace", "wrong": "経団連", "right": "経営層", "accept": False},
        ])

        data = yaml.safe_load((repo / "config/glossary-gemini.yaml").read_text())
        assert data["replacements"]["レブノル"] == "Revuno" and "MDクラウド" in data["protect"]
        assert "経団連" not in data["replacements"]
        assert result["pending"] == 0 and library.candidates("2026-09-14_1359")["items"] == []
        assert library.pushed == [("レブノル", "Revuno")]        # 守り札と見送りは タスク管理 に送らない
        assert (repo / "workspace/sessions/2026-09-14_1359/glossary_candidates.md").exists()

    def test_会議の外のフォルダは開かない(self, library):
        with pytest.raises(KeyError):
            library.candidates("../../config")


class TestVoices:
    def test_声の名前を直して外す(self, library, repo):
        from src.audio.voice_library import VoiceLibrary

        voices = VoiceLibrary(repo / "workspace/voices/library.json")
        voices.learn("s1", {"Headset Shuting Pan": [np.array([1.0, 0.0], dtype=np.float32)] * 6})
        voices.save()

        library.rename_voice("Headset Shuting Pan", "潘さん")
        assert [p["name"] for p in library.voices()["people"]] == ["潘さん"]

        library.forget_voice("潘さん")
        assert library.voices()["people"] == [] and library.changes == ["voices", "voices"]


def test_画面の口(library):
    app = FastAPI()
    app.include_router(build_library_router(library))
    client = TestClient(app)

    assert "会議アシスタント" in client.get("/library").text
    assert client.post("/api/library/dictionary/add", json={"dictionary": "gemini", "wrong": "a", "right": ""}).status_code == 400
    assert client.post("/api/library/dictionary/remove", json={"dictionary": "gemini", "wrong": "無い語"}).status_code == 404
    assert client.get("/api/library/candidates", params={"session": "2026-09-14_1359"}).json()["dictionary"] == "gemini"


def test_辞書の行を読み違えない(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text('replacements:\n  "a: b": "c"   # コロン入り\n  d: e\n', encoding="utf-8")

    glossary_edit.remove_replacement(path, "a: b", tmp_path / "trash")

    assert yaml.safe_load(path.read_text())["replacements"] == {"d": "e"}


class TestUnfinished:
    """会議のあと（仕上げの続き）— ブラウザを閉じたあとでも、ここから再開する。"""

    def _session(self, repo, name="2026-09-16_1400", **files):
        session = repo / "workspace" / "sessions" / name
        session.mkdir(parents=True, exist_ok=True)
        (session / "minutes_input.md").write_text("# 材料", encoding="utf-8")
        for key, body in files.items():
            (session / key.replace("__", ".")).write_text(body, encoding="utf-8")
        return session

    def test_残っている会議と残りの工程を出す(self, library, repo):
        self._session(repo)

        found = library.unfinished()

        assert found[0]["name"] == "2026-09-16_1400"
        assert [step["key"] for step in found[0]["steps"]] == ["finalize", "voices", "glossary", "minutes"]
        assert found[0]["running"] is False and found[0]["client"] is None

    def test_終わった会議は出さない(self, library, repo):
        from src import finish

        session = self._session(repo, transcripts_final__jsonl="{}", glossary_candidates__json="[]",
                                minutes__md="# 議事録")
        finish.clear_pending(session)
        (session / "voice_matches.jsonl").write_text("", encoding="utf-8")

        # 声の台帳だけはファイルで終わりを判定できないので残る＝会議は一覧に出る
        assert [step["key"] for step in library.unfinished()[0]["steps"]] == ["voices"]

    def test_押すと裏で走らせ二重には走らせない(self, library, repo, monkeypatch):
        self._session(repo)
        launched = []

        class FakeProcess:
            def poll(self):
                return None

        monkeypatch.setattr("src.ui.library.subprocess.Popen",
                            lambda argv, **kwargs: launched.append(argv) or FakeProcess())

        first = library.finish_session("2026-09-16_1400", "株式会社ミナト")
        second = library.finish_session("2026-09-16_1400")

        assert first["started"] is True and second["already"] is True
        assert len(launched) == 1 and launched[0][1].endswith("finish_meeting.py")
        assert library.unfinished()[0]["client"] == "株式会社ミナト"     # 相手も画面で選べる
        assert library.finish_status("2026-09-16_1400")["running"] is True

    def test_ログの終わりを画面に返す(self, library, repo):
        from src import finish

        session = self._session(repo)
        (session / finish.LOG_FILE).write_text("=== 仕上げを始めます\n作り直し 1/2\n", encoding="utf-8")

        assert library.finish_status("2026-09-16_1400")["lines"][-1] == "作り直し 1/2"

    def test_失敗して終わったら理由を画面に返す(self, library, repo, monkeypatch):
        """2026-09-24: ffmpeg が見つからず 1 秒で落ちたのに、画面が何も変わらなかった。"""
        from src import finish

        session = self._session(repo)

        class DeadProcess:
            def poll(self):
                return 1

        monkeypatch.setattr("src.ui.library.subprocess.Popen", lambda argv, **kwargs: DeadProcess())
        library.finish_session("2026-09-16_1400")
        with (session / finish.LOG_FILE).open("a", encoding="utf-8") as log:
            log.write("✗ 会議のあとの仕上げには ffmpeg が要ります。\n\nffmpeg が見つかりません。\n")

        status = library.finish_status("2026-09-16_1400")
        assert status["running"] is False and status["failed"] is True
        assert status["error"] == "✗ 会議のあとの仕上げには ffmpeg が要ります。"

    def test_走っていなければ失敗扱いにしない(self, library, repo):
        self._session(repo)
        assert library.finish_status("2026-09-16_1400")["failed"] is False


class TestMeetings:
    """過去の会議を振り返る（一覧・聞き直す・直す）。"""

    def _meeting(self, repo, name="2026-09-16_1000", final=False):
        session = repo / "workspace" / "sessions" / name
        session.mkdir(parents=True, exist_ok=True)
        rows = [{"speaker": "自分", "text": "おはようございます", "start_time": 0.0, "end_time": 2.0},
                {"speaker": "不明話者1", "text": "カタログナヒの件", "start_time": 5.0, "end_time": 9.0}]
        (session / "transcripts.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        if final:
            (session / "transcripts_final.jsonl").write_text(
                json.dumps({**rows[1], "text": "カタログナビの件"}, ensure_ascii=False), encoding="utf-8")
        (session / "audio").mkdir(exist_ok=True)
        (session / "audio" / "meeting.mp3").write_bytes(b"0" * 64)
        return session

    def test_一覧は新しい順で中身の見当が付く(self, library, repo):
        self._meeting(repo, "2026-09-10_1000")
        self._meeting(repo, "2026-09-16_1000")

        found = library.meetings()

        assert found[0]["name"] == "2026-09-16_1000"          # 名前順ではなく新しい順
        assert found[0]["lines"] == 2 and found[0]["tracks"] == ["meeting"]
        assert found[0]["speakers"] == ["不明話者1", "自分"]

    def test_作り直した全文があればそちらを出す(self, library, repo):
        self._meeting(repo, final=True)

        data = library.transcript("2026-09-16_1000")

        assert data["rebuilt"] is True
        assert data["rows"][0]["text"] == "カタログナビの件"

    def test_聞き直して直すと元の文も残る(self, library, repo):
        session = self._meeting(repo)

        result = library.edit_line("2026-09-16_1000", 5.0, text="カタログナビの件", speaker="MINATO南専務")

        assert result["before"]["text"] == "カタログナヒの件"
        rows = [json.loads(line) for line in (session / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows[1]["text"] == "カタログナビの件" and rows[1]["speaker"] == "MINATO南専務"
        assert rows[0]["text"] == "おはようございます"        # ほかの行は触らない
        edit = json.loads((session / "text_edits.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert edit["before"]["speaker"] == "不明話者1"

    def test_無い行や空の直しは断る(self, library, repo):
        self._meeting(repo)

        with pytest.raises(KeyError):
            library.edit_line("2026-09-16_1000", 999.0, text="x")
        with pytest.raises(ValueError):
            library.edit_line("2026-09-16_1000", 5.0)

    def test_無い音は返さない(self, library, repo):
        self._meeting(repo)

        assert library.audio_file("2026-09-16_1000", "meeting").name == "meeting.mp3"
        with pytest.raises(KeyError):
            library.audio_file("2026-09-16_1000", "self")


class TestMeetingDetail:
    """会議を選んだら、会議中の画面と同じものが出る（右に要約・決定事項・裏取り・議事録）。"""

    def _meeting(self, repo):
        session = repo / "workspace" / "sessions" / "2026-09-16_1000"
        session.mkdir(parents=True, exist_ok=True)
        (session / "transcripts.jsonl").write_text(json.dumps(
            {"speaker": "自分", "text": "確認します", "start_time": 0.0, "end_time": 2.0}, ensure_ascii=False),
            encoding="utf-8")
        (session / "state.json").write_text(json.dumps({
            "summary": "定例", "decisions": [{"text": "見積を出す", "owner": "自分"}],
            "todos": [{"text": "資料を送る", "by": "自分"}], "questions": ["納期は?"]}, ensure_ascii=False),
            encoding="utf-8")
        (session / "verifications.jsonl").write_text(json.dumps(
            {"verdict": "一致", "quote": "0.75 ドル", "note": "公式の価格表", "sources": ["https://例"]},
            ensure_ascii=False), encoding="utf-8")
        (session / "minutes_input.md").write_text("# 材料", encoding="utf-8")
        return session

    def test_要約と決定事項と裏取りとダウンロードが揃う(self, library, repo):
        self._meeting(repo)

        data = library.detail("2026-09-16_1000")

        assert data["state"]["decisions"][0] == {"text": "見積を出す", "owner": "自分", "status": "",
                                                 "due": "", "note": ""}
        assert data["state"]["todos"][0]["owner"] == "自分"      # by も owner として拾う
        assert data["state"]["questions"][0]["text"] == "納期は?"
        assert data["verifications"][0]["verdict"] == "一致"
        assert [file["kind"] for file in data["files"]] == ["input", "transcript_live"]
        assert data["rows"][0]["text"] == "確認します"           # 全文も同じ返事に入る

    def test_議事録が外にあるなら覚えて残りから外す(self, library, repo):
        session = self._meeting(repo)

        library.set_minutes_url("2026-09-16_1000", "https://app.notion.com/p/abc")

        assert library.detail("2026-09-16_1000")["minutes_url"] == "https://app.notion.com/p/abc"
        assert "minutes" not in [step["key"] for step in library.detail("2026-09-16_1000")["remaining"]]
        assert "登録済み" in json.loads((session / "finish_state.json").read_text(encoding="utf-8"))["minutes"]["note"]

    def test_URLでないものは覚えない(self, library, repo):
        self._meeting(repo)

        with pytest.raises(ValueError):
            library.set_minutes_url("2026-09-16_1000", "あとで貼る")

    def test_会議のフォルダの外は渡さない(self, library, repo):
        self._meeting(repo)

        with pytest.raises(KeyError):
            library.file_path("2026-09-16_1000", "settings")

    def test_声を覚えた会議は残りに出ない(self, library, repo):
        """台帳が正本（セッション側に印が無くても分かる）。「仕上げ残り 2」の中身が分からない、の直し。"""
        self._meeting(repo)
        voices = repo / "workspace" / "voices"
        voices.mkdir(parents=True, exist_ok=True)
        (voices / "library.json").write_text(json.dumps(
            {"people": {"MINATO南専務": [{"session": "2026-09-16_1000", "centroid": [1.0]}]}}), encoding="utf-8")

        assert "voices" not in [step["key"] for step in library.detail("2026-09-16_1000")["remaining"]]


class TestStartMeeting:
    """コックピットから会議を始める（立ち上げてすぐ会議、にしない）。"""

    def test_会議中ならその画面へ案内する(self, library, monkeypatch):
        monkeypatch.setattr(library, "meeting_status", lambda: {"running": True, "url": "http://127.0.0.1:8765/"})
        launched = []
        monkeypatch.setattr("src.app.meeting_launch.subprocess.Popen", lambda argv, **kwargs: launched.append(argv))

        result = library.start_meeting()

        assert result["already"] is True and launched == []

    def test_ターミナルを出さずに起こす(self, library, repo, monkeypatch):
        """2026-09-18 運用者 指摘「ターミナルは一切見えないようになるといいね」。

        端末が要らないことは確かめてある: 入力を待つ 3 か所はどれも画面が無いときだけ端末を使い、
        話者登録は skip_enrollment で飛ばしている。
        """
        monkeypatch.setattr(library, "meeting_status", lambda: {"running": False, "url": "http://127.0.0.1:8765/"})
        launched = []
        monkeypatch.setattr("src.app.meeting_launch.subprocess.Popen",
                            lambda argv, **kwargs: launched.append((argv, kwargs)))

        result = library.start_meeting()

        assert result["started"] is True
        argv, kwargs = launched[0]
        assert "Terminal" not in " ".join(argv)
        assert argv[1:] == ["-m", "src.main", "--mode", "meeting_loopback"]
        assert kwargs["start_new_session"] is True      # 画面を閉じても会議は続く

    def test_立ち上がらなかったときのために出力を残す(self, library, repo, monkeypatch):
        """切り離すと画面にも端末にも何も出ない。黙って死なせない。"""
        monkeypatch.setattr(library, "meeting_status", lambda: {"running": False, "url": ""})
        monkeypatch.setattr("src.app.meeting_launch.subprocess.Popen", lambda argv, **kwargs: None)

        library.start_meeting()

        assert (repo / "workspace" / "last_meeting.log").exists()


class TestFromTheList:
    """一覧から相手を決める・声を聞く（運用者 指摘 2026-09-16「一覧では何の会議か分からない」）。"""

    def _meeting(self, repo, name="2026-09-16_1000"):
        session = repo / "workspace" / "sessions" / name
        session.mkdir(parents=True, exist_ok=True)
        (session / "transcripts.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
            {"speaker": "MINATO南専務", "text": "テントの話です", "start_time": 10.0, "end_time": 15.0},
            {"speaker": "MINATO南専務", "text": "短い", "start_time": 20.0, "end_time": 21.0},
            {"speaker": "自分", "text": "はい", "start_time": 30.0, "end_time": 31.0}]), encoding="utf-8")
        (session / "audio").mkdir(exist_ok=True)
        (session / "audio" / "remote.mp3").write_bytes(b"0" * 64)
        return session

    def test_一覧から相手を決めて戻せる(self, library, repo, monkeypatch):
        self._meeting(repo)
        monkeypatch.setattr(library, "clients", lambda: [{"name": "株式会社ミナト", "client_id": "0045"}])

        library.set_client("2026-09-16_1000", "株式会社ミナト")
        assert library.meetings()[0]["client"] == "株式会社ミナト"

        library.set_client("2026-09-16_1000", "")      # 空にしたら自社へ戻す
        assert library.meetings()[0]["client"] is None

    def test_声を聞く用に長めの発言を選ぶ(self, library, repo):
        from src.audio.voice_library import VoiceLibrary

        session = self._meeting(repo)
        voices = VoiceLibrary(repo / "workspace/voices/library.json")
        voices.learn(session.name, {"MINATO南専務": [np.array([1.0, 0.0], dtype=np.float32)] * 6})
        voices.save()

        sample = library.voice_sample("MINATO南専務")

        assert sample["start"] == 10.0 and sample["track"] == "remote"   # 短い行は選ばない
        assert sample["text"] == "テントの話です"

    def test_録音が無ければ聞けないと言う(self, library, repo):
        from src.audio.voice_library import VoiceLibrary

        session = self._meeting(repo)
        (session / "audio" / "remote.mp3").unlink()
        voices = VoiceLibrary(repo / "workspace/voices/library.json")
        voices.learn(session.name, {"MINATO南専務": [np.array([1.0, 0.0], dtype=np.float32)] * 6})
        voices.save()

        with pytest.raises(KeyError):
            library.voice_sample("MINATO南専務")


class TestGroupingAndDelete:
    """話題ごとのまとまり・参加者・会議の削除（運用者 指摘 2026-09-16）。"""

    def _meeting(self, repo, name="2026-09-16_1000"):
        session = repo / "workspace" / "sessions" / name
        session.mkdir(parents=True, exist_ok=True)
        (session / "transcripts.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
            {"speaker": "自分", "text": "はい", "start_time": 0.0, "end_time": 4.0},
            {"speaker": "MINATO南専務", "text": "口コミの件", "start_time": 10.0, "end_time": 70.0}]),
            encoding="utf-8")
        (session / "state.json").write_text(json.dumps({
            "topics": [{"id": "P1", "text": "挨拶", "created_at": 0.0, "status": "closed"},
                       {"id": "P2", "text": "口コミ対応", "created_at": 100.0, "status": "active"}],
            "decisions": [{"text": "返信で対応する", "by": "自分", "created_at": 120.0},
                          {"text": "会議前の決定", "by": "自分", "created_at": -5.0}],
            "todos": [{"text": "返信文を作る", "by": "自分", "due": "2026-09-20", "created_at": 130.0}],
            "questions": [{"text": "削除申請は通るか", "created_at": 110.0, "note": "通らない前提"}]},
            ensure_ascii=False), encoding="utf-8")
        return session

    def test_話題ごとに決定とTODOと質問をまとめる(self, library, repo):
        self._meeting(repo)

        sections = library.detail("2026-09-16_1000")["sections"]

        assert [section["topic"] for section in sections] == ["（話題の前・どこにも入らないもの）", "口コミ対応"]
        assert sections[1]["decisions"][0]["text"] == "返信で対応する"
        assert sections[1]["questions"][0]["note"] == "通らない前提"
        assert sections[1]["todos"][0]["due"] == "2026-09-20"     # 期限も出す（表にするため）

    def test_参加者は話した量の多い順(self, library, repo):
        self._meeting(repo)

        people = library.detail("2026-09-16_1000")["participants"]

        assert [person["name"] for person in people] == ["MINATO南専務", "自分"]
        assert people[0]["minutes"] == 1 and people[0]["lines"] == 1

    def test_会議は消さずにゴミ箱へ退避する(self, library, repo):
        self._meeting(repo, "2026-09-16_1000")
        self._meeting(repo, "2026-09-16_1100")

        result = library.delete_meetings(["2026-09-16_1000", "2026-09-16_1100"])

        assert sorted(result["moved"]) == ["2026-09-16_1000", "2026-09-16_1100"]
        assert not (repo / "workspace/sessions/2026-09-16_1000").exists()
        moved = Path(result["trash"])
        assert (moved / "2026-09-16_1000" / "state.json").exists()      # 中身ごと残る
        assert "戻すには" in (moved / "WHY.md").read_text(encoding="utf-8")

    def test_無い会議や空の指定は断る(self, library, repo):
        with pytest.raises(ValueError):
            library.delete_meetings([])
        with pytest.raises(KeyError):
            library.delete_meetings(["../config"])


class TestRegisterMinutes:
    """議事録を画面から Notion（タスク管理の議事録 DB）へ流し込む。

    運用者 依頼 2026-09-17「流し込む。議事録 WF の仕様に沿って。押してから（安定したら自動化）」。
    """

    def _session(self, repo):
        session = repo / "workspace" / "sessions" / "2026-09-14_1359"
        (session / "minutes.md").write_text("# 議事録: 見積\n\n## 決定事項\n- 出す\n", encoding="utf-8")
        (session / "client.json").write_text(json.dumps({"name": "取引先A"}, ensure_ascii=False), encoding="utf-8")
        return session

    def test_押すと登録し議事録のURLを覚える(self, library, repo, monkeypatch):
        from src import finish, task_hub_minutes

        session = self._session(repo)
        called = {}

        def fake_register(session_dir, *, client_name, meeting_date, shared_dir, **kwargs):
            called.update(session=session_dir, client=client_name, date=meeting_date, **kwargs)
            return task_hub_minutes.Registration(url="https://notion.so/p1", page_id="p1", linked=2,
                                               chains={"cards": 3, "facts": 1, "notes": []})

        monkeypatch.setattr(task_hub_minutes, "register", fake_register)

        result = library.register_minutes("2026-09-14_1359")

        assert result["url"] == "https://notion.so/p1" and result["linked"] == 2
        assert result["chains"] == {"cards": 3, "facts": 1, "notes": []}
        assert called["client"] == "取引先A" and called["date"] == "2026-09-14"
        assert called["prompts"].name == "prompts"          # 抽出のプロンプトを渡している
        assert finish.read_state(session)["minutes"]["url"] == "https://notion.so/p1"

    def test_登録できない理由はそのまま画面へ(self, repo, monkeypatch):
        from src import task_hub_minutes

        self._session(repo)
        library = Library(repo, settings)
        app = FastAPI()
        app.include_router(build_library_router(library))
        monkeypatch.setattr(task_hub_minutes, "register", lambda *a, **k: (_ for _ in ()).throw(
            task_hub_minutes.NotionMinutesError("議事録 DB が設定されていません")))

        response = TestClient(app).post("/api/library/minutes/register", json={"session": "2026-09-14_1359"})

        assert response.status_code == 502 and "議事録 DB" in response.json()["detail"]

    def test_登録済みなら二度押しても作らない(self, library, repo, monkeypatch):
        """押したあとにスキルを回すとページが 2 つできる。まず二度押しを止める。"""
        from src import finish, task_hub_minutes

        session = self._session(repo)
        finish.mark(session, "minutes", note="登録済み", url="https://notion.so/p1")
        monkeypatch.setattr(task_hub_minutes, "register",
                            lambda *a, **k: pytest.fail("登録済みなのに呼ばれた"))

        with pytest.raises(ValueError, match="登録済み"):
            library.register_minutes("2026-09-14_1359")


class TestPersonas:
    """画面から「話者名 → Notion のペルソナ名」を直す（運用者 依頼 2026-09-17）。"""

    def _settings(self, repo):
        (repo / "config" / "settings.yaml").write_text(
            "task_hub:\n  client_name: 自社\n  personas:\n    自分: 自社社長\n\n"
            "meeting:\n  self_name: 自分\n", encoding="utf-8")

    def test_会議に出た名前と対応表を並べる(self, repo):
        self._settings(repo)
        session = repo / "workspace" / "sessions" / "2026-09-14_1359"
        (session / "transcripts.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
            {"speaker": "自分", "text": "あ"}, {"speaker": "参加者A", "text": "い"},
            {"speaker": "不明話者3", "text": "う"}]), encoding="utf-8")
        library = Library(repo, lambda: {"meeting": {"self_name": "自分"}})

        result = library.personas()

        names = {person["speaker"]: person for person in result["people"]}
        assert names["自分"]["persona"] == "自社社長" and names["自分"]["self"] is True
        assert names["参加者A"]["persona"] == ""            # 未設定は空で出す（登録されない）
        assert "不明話者3" not in names                      # 不明話者は出さない

    def test_書き換えると設定に残る(self, repo):
        self._settings(repo)
        library = Library(repo, lambda: {"meeting": {"self_name": "自分"}})

        result = library.set_persona("参加者A", "さくら歯科 院長")

        assert result["persona"] == "さくら歯科 院長"
        assert library.personas()["people"][0]["speaker"]  # 一覧が壊れていない
        assert "さくら歯科 院長" in (repo / "config" / "settings.yaml").read_text(encoding="utf-8")


class TestMinutesPanel:
    """登録済みの会議では、URL を貼る欄を出さない（運用者 指摘 2026-09-17「覚えるって何？」）。"""

    def test_登録したときの覚え書きも画面へ返す(self, library, repo):
        from src import finish

        session = repo / "workspace" / "sessions" / "2026-09-14_1359"
        (session / "transcripts.jsonl").write_text(
            json.dumps({"speaker": "自分", "text": "あ", "start_time": 0, "end_time": 1}, ensure_ascii=False),
            encoding="utf-8")
        finish.mark(session, "minutes", note="さくら歯科 の議事録 DB に登録", url="https://notion.so/p1")

        detail = library.detail("2026-09-14_1359")

        assert detail["minutes_url"] == "https://notion.so/p1"
        assert detail["minutes_note"] == "さくら歯科 の議事録 DB に登録"

    def test_画面は登録済みかどうかで出し分ける(self):
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/library.html").read_text(encoding="utf-8")

        assert "別の場所に付け替える" in html            # 登録済みのときの逃げ道
        assert "URL を貼って「覚える」" in html          # 手で貼るほうの説明

    def test_つないでいなければ登録ボタンを出さない(self):
        """押しても動かないボタンを出さない（2026-09-21 に発見）。

        配布版は `notion_tasks: false` なのに「Notion に登録」だけ出ていた。
        押すと、作者の社内ライブラリを読みにいって失敗する。
        """
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/library.html").read_text(encoding="utf-8")
        assert "data.can_register_minutes" in html

    def test_詳細は登録できるかを返す(self, repo):
        """設定でつないでいるときだけ true。"""
        session = repo / "workspace" / "sessions" / "2026-09-14_1359"
        session.mkdir(parents=True, exist_ok=True)
        (session / "transcripts.jsonl").write_text("", encoding="utf-8")
        library = Library(repo, lambda: {"task_hub": {"notion_tasks": False}},
                          sessions_dir=repo / "workspace" / "sessions")
        assert library.detail("2026-09-14_1359")["can_register_minutes"] is False
        library = Library(repo, lambda: {"task_hub": {"notion_tasks": True}},
                          sessions_dir=repo / "workspace" / "sessions")
        assert library.detail("2026-09-14_1359")["can_register_minutes"] is True


class TestRenameSpeaker:
    """会議のあとに不明話者へ名前を付ける（運用者 指摘 2026-09-17「不明話者3 はどこにもない」）。"""

    def _session(self, repo):
        session = repo / "workspace" / "sessions" / "2026-09-14_1359"
        rows = [{"speaker": "不明話者3", "text": "はい", "start_time": 0, "end_time": 2},
                {"speaker": "自分", "text": "どうも", "start_time": 2, "end_time": 4},
                {"speaker": "不明話者3", "text": "よろしく", "start_time": 4, "end_time": 6}]
        (session / "transcripts.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        (session / "transcripts_final.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return session

    def test_全文の話者名を書き換えて記録を残す(self, library, repo):
        session = self._session(repo)

        result = library.rename_speaker("2026-09-14_1359", "不明話者3", "KDC院長")

        assert result["lines"] == 2
        for name in ("transcripts.jsonl", "transcripts_final.jsonl"):
            text = (session / name).read_text(encoding="utf-8")
            assert "不明話者3" not in text and text.count("KDC院長") == 2
        record = json.loads((session / "renames.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert record["old"] == "不明話者3" and record["new"] == "KDC院長"

    def test_不明話者という名前は付けられない(self, library, repo):
        self._session(repo)

        with pytest.raises(ValueError, match="不明話者"):
            library.rename_speaker("2026-09-14_1359", "不明話者3", "不明話者4")

    def test_その会議にいない人は断る(self, library, repo):
        self._session(repo)

        with pytest.raises(KeyError):
            library.rename_speaker("2026-09-14_1359", "居ない人", "誰か")

    def test_画面に名前を付けるボタンがある(self):
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/library.html").read_text(encoding="utf-8")

        assert "名前を付ける" in html and "/api/library/speaker/rename" in html

    def test_行を指定するとその行だけ直す(self, library, repo):
        """1 つの「不明話者」に複数人が混ざっていることがある（運用者 指摘 2026-09-17）。"""
        session = self._session(repo)

        result = library.rename_speaker("2026-09-14_1359", "不明話者3", "KDC院長", starts=[4.0])

        assert result["lines"] == 1
        rows = [json.loads(line) for line in
                (session / "transcripts_final.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [row["speaker"] for row in rows] == ["不明話者3", "自分", "KDC院長"]

    def test_覚えなかった理由も返す(self, library, repo):
        """「声がそろわない＝複数人が混ざっている可能性」を黙って捨てない。"""
        self._session(repo)

        result = library.rename_speaker("2026-09-14_1359", "不明話者3", "KDC院長")

        assert "skipped" in result and isinstance(result["skipped"], dict)

    def test_画面に範囲で名前を付ける道具がある(self):
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/library.html").read_text(encoding="utf-8")

        assert "ここから（" in html and "starts" in html and "複数人が混ざっている" in html


class TestSettingsTab:
    """機能の ON/OFF と各種の設定を画面から（運用者 依頼 2026-09-17）。"""

    def _settings(self, repo):
        (repo / "config" / "settings.yaml").write_text(
            "meeting:\n  self_name: 自分\n  screen_capture:\n    enabled: false   # 既定 off\n"
            "  voice_library:\n    enabled: true\n", encoding="utf-8")

    def test_いまの値つきで一覧を返す(self, repo):
        self._settings(repo)
        library = Library(repo, settings)

        result = library.settings_fields()

        keys = {field["key"]: field for field in result["fields"]}
        assert keys["meeting.screen_capture.enabled"]["value"] is False
        assert result["path"].endswith("settings.yaml")

    def test_押すと設定が書き換わり元のファイルは退避される(self, repo):
        self._settings(repo)
        library = Library(repo, settings)

        assert library.set_setting("meeting.screen_capture.enabled", True)["value"] is True

        text = (repo / "config" / "settings.yaml").read_text(encoding="utf-8")
        assert "enabled: true" in text and "既定 off" in text
        assert list((repo / "workspace" / "99-trash" / "settings").glob("*settings.yaml"))

    def test_画面に設定タブがある(self):
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/library.html").read_text(encoding="utf-8")

        assert 'data-tab="settings"' in html and "/api/library/settings" in html
        assert "外へ出る" in html          # 音声が外へ出る設定には印を付ける

    def test_動かし方と端末の能力も返す(self, repo, monkeypatch):
        """できない動かし方は、押せない形で出す（会議中に足りないと分かるのを避ける）。"""
        from src import capability

        self._settings(repo)
        monkeypatch.setattr(capability, "look", lambda settings=None: capability.Capability(
            apple_silicon=True, memory_gb=36, ollama=False, ollama_models=[],
            claude_cli=True, gemini_key=True, deepgram_key=True))
        library = Library(repo, settings)

        result = library.settings_fields()

        presets = {one["name"]: one for one in result["presets"]}
        assert result["capability"]["offline"] is False
        assert presets["quality"]["usable"] is True and presets["secret"]["usable"] is False
        assert result["why_not_offline"]

    def test_画面に動かし方の選択がある(self):
        html = (Path(__file__).resolve().parent.parent / "src/ui/static/library.html").read_text(encoding="utf-8")

        assert "data-preset=" in html and "/api/library/settings/preset" in html
        assert "オフラインでも会議を通せます" in html


class TestSelfNameRename:
    """2026-09-18 運用者 報告: 自分の名前を変えたら、哲学カードの人物が 2 つになった。

    古い会議の記録には前の名前が残るので、候補を集めると両方が並ぶ。
    同じ人だと機械には分からないので、変えた側で覚えておく。
    """

    def _prepare(self, repo, personas: str = "") -> Library:
        (repo / "config" / "settings.yaml").write_text(
            f"task_hub:\n  client_name: 自社\n  personas:\n{personas}\n"
            "meeting:\n  self_name: 自分\n", encoding="utf-8")
        session = repo / "workspace" / "sessions" / "2026-09-14_1359"
        (session / "transcripts.jsonl").write_text("\n".join(
            json.dumps(row, ensure_ascii=False) for row in
            [{"speaker": "自分", "text": "あ"}, {"speaker": "LEI長谷川", "text": "い"}]), encoding="utf-8")
        return Library(repo, lambda: {"meeting": {"self_name": "自分"}})

    def test_名前を変えると前の名前を覚える(self, repo):
        library = self._prepare(repo)

        library.set_setting("meeting.self_name", "自社自分")

        assert library._previous_self_names() == {"自分"}

    def test_前の名前は哲学カードの候補に出ない(self, repo):
        library = self._prepare(repo)
        library.set_setting("meeting.self_name", "自社自分")

        names = [one["speaker"] for one in library.personas()["people"]]

        assert "自分" not in names
        assert "LEI長谷川" in names

    def test_対応表に載っている名前は残す(self, repo):
        """哲学カードの行き先を決めてある名前を黙って消さない。"""
        library = self._prepare(repo, personas="    自分: 自社社長\n")
        library.set_setting("meeting.self_name", "自社自分")

        names = [one["speaker"] for one in library.personas()["people"]]

        assert "自分" in names

    def test_同じ名前に変えても覚えない(self, repo):
        library = self._prepare(repo)

        library.set_setting("meeting.self_name", "自分")

        assert library._previous_self_names() == set()
