"""画面共有の取り込み（何のページの話かを、あとから辿れるように）。"""

from __future__ import annotations

import json
import subprocess

from src.screen.capture import (
    SCREENS_FILE,
    ScreenConfig,
    ScreenWatcher,
    difference,
    find_window,
    is_meeting_url,
    pages,
    pick_page,
    shown_urls,
)

WINDOWS = [
    {"id": 1, "owner": "Slack", "title": "チャンネル", "width": 1800, "height": 1130},
    {"id": 2, "owner": "zoom.us", "title": "Zoom ミーティング", "width": 1600, "height": 900},
    {"id": 3, "owner": "zoom.us", "title": "", "width": 1600, "height": 900},
    {"id": 4, "owner": "Google Chrome", "title": "見積書", "width": 2000, "height": 1200},
]


class TestFindWindow:
    def test_会議のウィンドウを選ぶ(self):
        """画面全体は撮らない。関係ないアプリ（Slack・メール）も撮らない。"""
        window = find_window(ScreenConfig(), lister=lambda: WINDOWS)

        assert window["id"] == 2      # ブラウザより大きくても、会議のウィンドウを優先する

    def test_名前の無い窓や小さい窓は撮らない(self):
        small = [{"id": 9, "owner": "zoom.us", "title": "通知", "width": 200, "height": 80}]

        assert find_window(ScreenConfig(), lister=lambda: small) is None
        assert find_window(ScreenConfig(), lister=lambda: [WINDOWS[2]]) is None

    def test_対象のアプリが居なければ何もしない(self):
        assert find_window(ScreenConfig(), lister=lambda: [WINDOWS[0]]) is None


def test_変わった画素の割合で見る():
    """平均の差だと、白地の資料では別のページでも 0.007 にしかならなかった（実測）。"""
    before = bytes([255] * 100)
    same = bytes([255] * 100)
    scrolled = bytes([255] * 90 + [0] * 10)

    assert difference(before, same) == 0.0
    assert difference(before, scrolled) == 0.1
    assert difference(before, b"") == 1.0        # 取れなかったときは「別物」に倒す


def test_URLとページ名を取り出す():
    lines = [{"text": "https://www.example.co.jp/catalog-navi/ を見てください", "h": 0.02},
             {"text": "カタログナビ — 比較", "h": 0.05},
             {"text": "お問い合わせは example.co.jp まで", "h": 0.02}]

    page = pick_page(lines)

    assert page["urls"] == ["https://www.example.co.jp/catalog-navi/"]
    assert page["title"] == "カタログナビ — 比較"      # いちばん大きい文字＝見出し
    assert page["domains"] == ["example.co.jp"]


class TestWatcher:
    def _watcher(self, tmp_path, monkeypatch, *, frames):
        config = ScreenConfig(enabled=True, ocr=False, change_threshold=0.02)
        watcher = ScreenWatcher(tmp_path, config, None)
        watcher._meeting_clock = lambda: 12.0
        monkeypatch.setattr("src.screen.capture.find_window", lambda config, **kwargs: {"id": 2, "owner": "zoom.us", "title": "Zoom"})
        def fake_capture(window_id, target, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"jpg")
            return True

        monkeypatch.setattr("src.screen.capture.capture_window", fake_capture)
        marks = iter(frames)
        monkeypatch.setattr("src.screen.capture.fingerprint", lambda path, **kwargs: next(marks))
        return watcher

    def test_変わったときだけ残す(self, tmp_path, monkeypatch):
        watcher = self._watcher(tmp_path, monkeypatch, frames=[bytes([0] * 100), bytes([0] * 100), bytes([255] * 100)])

        first = watcher.tick()
        same = watcher.tick()
        changed = watcher.tick()

        assert first["file"] == "screens/000012.jpg" and same is None and changed is not None
        assert watcher.shots == 2 and watcher.skipped == 1
        lines = (tmp_path / SCREENS_FILE).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and json.loads(lines[0])["at"] == 12.0

    def test_会議のウィンドウが無ければ何もしない(self, tmp_path, monkeypatch):
        watcher = self._watcher(tmp_path, monkeypatch, frames=[b""])
        monkeypatch.setattr("src.screen.capture.find_window", lambda config, **kwargs: None)

        assert watcher.tick() is None and watcher.shots == 0

    def test_切ってあれば動かさない(self, tmp_path):
        watcher = ScreenWatcher(tmp_path, ScreenConfig(enabled=False), None)

        watcher.start(lambda: 0.0)

        assert watcher._thread is None


def test_見せられたページを1件にまとめる(tmp_path):
    (tmp_path / SCREENS_FILE).write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in [
        {"at": 10.0, "file": "screens/000010.jpg", "urls": ["https://example.jp/a"], "title": "比較"},
        {"at": 40.0, "file": "screens/000040.jpg", "urls": ["https://example.jp/a"], "title": ""},
        {"at": 90.0, "file": "screens/000090.jpg", "urls": ["https://example.jp/b"], "title": "見積"}]),
        encoding="utf-8")

    found = pages(tmp_path)

    assert [entry["url"] for entry in found] == ["https://example.jp/a", "https://example.jp/b"]
    assert found[0]["shots"] == 2 and found[0]["title"] == "比較" and found[0]["first_at"] == 10.0


class TestWhichWindow:
    """撮るのは「会議の窓」だけ（運用者 指摘 2026-09-17）。

    ブラウザは会議にも、この道具にも、調べ物にも使う。大きい窓を選ぶだけでは、
    会議中の調べ物や、この会議アシスタントの画面（クライアント名が出ている）を撮ってしまう。
    """

    def rows(self, *windows):
        return lambda: [{"id": index, "owner": owner, "title": title, "width": width, "height": 900}
                        for index, (owner, title, width) in enumerate(windows, start=1)]

    def test_ブラウザは会議の目印が無ければ撮らない(self):
        config = ScreenConfig(enabled=True)
        lister = self.rows(("Comet", "エアフロー 歯科 効果 - Google 検索", 2560),
                           ("Google Chrome", "会議アシスタント", 1800))

        assert find_window(config, lister=lister) is None

    def test_Zoomの窓は目印が無くても撮る(self):
        config = ScreenConfig(enabled=True)
        lister = self.rows(("Comet", "調べ物", 2560), ("zoom.us", "ズーム", 1400))

        assert find_window(config, lister=lister)["owner"] == "zoom.us"

    def test_GoogleMeetのタブは撮る(self):
        config = ScreenConfig(enabled=True)
        lister = self.rows(("Comet", "調べ物のページ", 2560),
                           ("Comet", "Meet - abc-defg-hij", 1600))

        assert find_window(config, lister=lister)["title"].startswith("Meet")

    def test_自分の画面は撮り返さない(self):
        """会議アシスタントにはクライアント名や過去の会議が出ている。"""
        config = ScreenConfig(enabled=True)
        lister = self.rows(("Google Chrome", "会議アシスタント – Google Meet", 2000))

        assert find_window(config, lister=lister) is None

    def test_会議の窓が複数なら大きいほうを撮る(self):
        config = ScreenConfig(enabled=True)
        lister = self.rows(("zoom.us", "Zoom ミーティング", 1200), ("Comet", "Google Meet", 2400))

        assert find_window(config, lister=lister)["width"] == 2400


class TestMeetingKey:
    """どの会議だったかの手がかり（同じ会議を 1 つにまとめる下ごしらえ・PLAN の ④）。"""

    def test_MeetのコードとZoomのIDを拾う(self):
        from src.screen.capture import meeting_key

        assert meeting_key("Meet - abc-defg-hij") == {"kind": "meet", "key": "abc-defg-hij"}
        assert meeting_key("https://us02web.zoom.us/j/81234567890?pwd=x") == {
            "kind": "zoom", "key": "81234567890"}

    def test_画面から読んだIDの空白は詰める(self):
        """Zoom は「812 3456 7890」と出す。同じ会議だと分かる形にそろえる。"""
        from src.screen.capture import meeting_key

        assert meeting_key("ミーティング ID: 812 3456 7890")["key"] == "81234567890"

    def test_手がかりが無ければ何も残さない(self):
        from src.screen.capture import meeting_key

        assert meeting_key("見積書.pdf", "https://example.com/a") is None

    def test_一度だけ残す(self, tmp_path, monkeypatch):
        from src.screen import capture

        watcher = capture.ScreenWatcher(tmp_path, capture.ScreenConfig(enabled=True, ocr=False), None)
        watcher._remember_meeting({"at": 12.0, "window": "Meet - abc-defg-hij", "urls": []})
        watcher._remember_meeting({"at": 30.0, "window": "Meet - zzz-zzzz-zzz", "urls": []})

        import json

        saved = json.loads((tmp_path / capture.MEETING_KEY_FILE).read_text(encoding="utf-8"))
        assert saved["key"] == "abc-defg-hij" and saved["at"] == 12.0


class TestMeetingOwnUrl:
    """2026-09-18 運用者 指摘「見せられたページが、GoogleMeet だと画面の URL で意味がない」。

    ブラウザのアドレス欄を読んでいるので、共有されている中身とは関係ない住所が入る。
    """

    def test_会議アプリ自身の住所を見分ける(self):
        assert is_meeting_url("https://meet.google.com/abc-defg-hij?pli=1") is True
        assert is_meeting_url("https://your-org.zoom.us/j/123") is True
        assert is_meeting_url("https://example.com/plan") is False

    def test_見せられたページから会議アプリを外す(self, tmp_path):
        rows = [
            {"at": 6.5, "file": "screens/1.jpg", "title": "Meet",
             "urls": ["https://meet.google.com/abc-defg-hij?pli=1"]},
            {"at": 40.0, "file": "screens/2.jpg", "title": "見積書",
             "urls": ["https://meet.google.com/abc-defg-hij?pli=1", "https://example.com/quote"]},
        ]
        (tmp_path / "screens.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

        found = pages(tmp_path)
        urls = [one["url"] for one in found]

        assert "https://example.com/quote" in urls
        assert not any("meet.google.com" in (url or "") for url in urls)

    def test_会議アプリしか写っていなければ見出しでまとめる(self):
        """URL が無くなっても、控えごとバラバラにしない。"""
        entry = {"urls": ["https://meet.google.com/abc-defg-hij"]}

        assert shown_urls(entry) == []
