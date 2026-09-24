"""メニューバーに居座る本体（PyObjC）。

新しい依存を足していない: `pyobjc-framework-Quartz` が入っているので AppKit は既にある。

このアプリは**会議を動かさない**。会議を始めるときは、いままでどおり
  `scripts/会議を始める.command` をターミナルで開く（起動セルフチェックの確認と
  話者登録が入力を受けるため。2026-09-16 に決めた形をそのまま使う）。
"""

from __future__ import annotations

import subprocess
import threading
import time
import webbrowser
from pathlib import Path

from src.app import login_item, meeting_launch, running

REFRESH_SEC = 5.0
"""状態を見に行く間隔。ポートを覗くだけなので会議の負荷にはならない。"""


def start_meeting(repo: Path, python: Path, on_trouble=None) -> None:
    """会議を始める。ターミナルは出さない（`src/app/meeting_launch.py` の注記）。

    画面が立つまで見張り、立たなければ起動ログを見せる（切り離すと何も出ないため）。
    """
    process = meeting_launch.start(repo, python)
    threading.Thread(target=_watch_start, args=(process, repo, on_trouble), daemon=True).start()


def _watch_start(process, repo: Path, on_trouble) -> None:
    """画面が立つのを待って開く。立たなければ理由を渡す。"""
    deadline = time.monotonic() + meeting_launch.WAIT_SEC
    while time.monotonic() < deadline:
        meeting = running.find_meeting()
        if meeting is not None:
            webbrowser.open(meeting.url)
            return
        if process.poll() is not None:
            break                            # 先に死んだ。待っても立たない
        time.sleep(1.0)
    if on_trouble is not None:
        on_trouble(meeting_launch.why_not_started(process, repo))


def show_trouble(message: str) -> None:      # pragma: no cover - 画面が要る
    """困りごとをそのまま見せる。常駐アプリには出す場所がないので、macOS の警告で出す。"""
    subprocess.run(["osascript", "-e",
                    'display alert "会議を始められませんでした" message %s' % _applescript(message)],
                   capture_output=True)


def _applescript(text: str) -> str:
    """AppleScript の文字列にする（引用符と改行を潰す）。"""
    return '"%s"' % text.replace("\\", "\\\\").replace('"', "'").replace("\n", "\\n")


def open_library(repo: Path, python: Path) -> None:
    """会議アシスタントを開く。もう開いていれば、そちらを開くだけ。"""
    if running.port_open(running.LIBRARY_PORT):
        webbrowser.open(f"http://{running.HOST}:{running.LIBRARY_PORT}/library")
        return
    subprocess.Popen([str(python), str(Path(repo) / "scripts" / "library_ui.py")])


def run(repo: Path, python: Path) -> int:            # pragma: no cover - 画面が要る
    """常駐を始める。終わるまで戻らない。"""
    import AppKit
    import objc
    from PyObjCTools import AppHelper

    repo, python = Path(repo), Path(python)
    home = Path.home()

    class Delegate(AppKit.NSObject):
        def init(self):
            self = objc.super(Delegate, self).init()
            if self is None:
                return None
            self.meeting = None
            bar = AppKit.NSStatusBar.systemStatusBar()
            self.item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            self.menu = AppKit.NSMenu.alloc().init()
            self.menu.setAutoenablesItems_(False)
            self.item.setMenu_(self.menu)
            self.refresh_(None)
            AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                REFRESH_SEC, self, "refresh:", None, True)
            return self

        # ---------------------------------------------------------- 描き直す

        def refresh_(self, _timer):
            self.meeting = running.find_meeting()
            self.item.button().setTitle_(running.title(self.meeting))
            self.menu.removeAllItems()
            self._row(running.status_line(self.meeting), None)
            self.menu.addItem_(AppKit.NSMenuItem.separatorItem())
            if self.meeting is None:
                self._row("会議を始める…", "startMeeting:")
            else:
                self._row("会議の画面を開く", "openMeeting:")
            self._row("会議アシスタントを開く", "openLibrary:")
            self.menu.addItem_(AppKit.NSMenuItem.separatorItem())
            self._row("ログインしたら起動する", "toggleLogin:",
                      check=login_item.is_enabled(home))
            self.menu.addItem_(AppKit.NSMenuItem.separatorItem())
            self._row("終了", "quit:")

        # PyObjC は NSObject の中のメソッドを全部 Objective-C の口として扱う。
        #   ただの手伝いには印を付ける（付けないと BadPrototypeError で起動できない）。
        @objc.python_method
        def _row(self, label, action, check=False):
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, action, "")
            item.setEnabled_(action is not None)
            if action is not None:
                item.setTarget_(self)
            if check:
                item.setState_(AppKit.NSControlStateValueOn)
            self.menu.addItem_(item)

        # ------------------------------------------------------------ 押された

        def startMeeting_(self, _sender):
            start_meeting(repo, python, on_trouble=show_trouble)

        def openMeeting_(self, _sender):
            if self.meeting is not None:
                webbrowser.open(self.meeting.url)

        def openLibrary_(self, _sender):
            open_library(repo, python)

        def toggleLogin_(self, _sender):
            if login_item.is_enabled(home):
                login_item.disable(home)
            else:
                login_item.enable(_app_path(), home)
            self.refresh_(None)

        def quit_(self, _sender):
            AppKit.NSApplication.sharedApplication().terminate_(self)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)   # Dock に出さない
    delegate = Delegate.alloc().init()
    app.setDelegate_(delegate)
    AppHelper.runEventLoop()
    return 0


def _app_path() -> Path:                             # pragma: no cover - 画面が要る
    """自分が入っている .app の場所。見つからなければ /Applications を当てにする。"""
    import AppKit

    path = AppKit.NSBundle.mainBundle().bundlePath()
    if path and str(path).endswith(".app"):
        return Path(str(path))
    return Path("/Applications/会議アシスタント.app")
