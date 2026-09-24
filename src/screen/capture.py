"""会議中に**相手が画面共有している内容**を控える（何のページの話かを、あとから辿れるように）。

2026-09-16 運用者 依頼: 「画面共有されていると、相手はサイト名や URL を読み上げない。
あとでそのページを探すのが大変。キャプチャを会議の発言の周りに置いて、URL も記録してほしい」。

やること:

1. **Zoom のウィンドウだけ**を一定間隔で取り込む（`screencapture -l <窓>`）。画面全体は撮らない
2. 前の 1 枚と**見た目が変わったときだけ**残す（同じ画面を何十枚も残さない）
3. 文字を読み取り（`src/screen/ocr.swift` を別プロセスで）、**URL とページ名らしき行**を取り出す
4. `screens/<秒>.jpg` と `screens.jsonl` に残す。画面の詳細では発言の隣に出し、議事録の材料にも並べる

落ちても会議は止めない: 文字の読み取りは**別プロセス**（pyobjc から Vision を呼ぶと Python ごと落ちた・
2026-09-16 に 5 回連続で確認）。取り込みも読み取りも、失敗したらその 1 枚を諦める。
外へは何も送らない（取り込みも読み取りも手元で完結）。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SCREENS_DIR = "screens"
SCREENS_FILE = "screens.jsonl"
MEETING_KEY_FILE = "meeting_key.json"
"""どの会議だったかの手がかり（同じ会議に複数人が入ったとき、1 つにまとめるための下ごしらえ）。"""

MEETING_KEYS = (
    ("meet", re.compile(r"meet\.google\.com/([a-z]{3}-[a-z]{4}-[a-z]{3})", re.I)),
    ("meet", re.compile(r"\bMeet\s*[-–]\s*([a-z]{3}-[a-z]{4}-[a-z]{3})\b", re.I)),
    ("zoom", re.compile(r"zoom\.us/j/(\d{9,12})")),
    ("zoom", re.compile(r"(?:ミーティング\s*ID|Meeting\s*ID)[:：]?\s*([\d\s]{9,16})", re.I)),
    ("teams", re.compile(r"teams\.microsoft\.com/l/meetup-join/([\w%./-]{10,})", re.I)),
)
"""会議を見分ける手がかり（強い順）。タイトルと、画面から読んだ URL・文字の両方を見る。"""
URL_PATTERN = re.compile(r"https?://[\w./?=&%:#@~+,;!$'()*\[\]-]+")
DOMAIN_PATTERN = re.compile(r"\b(?:[\w-]+\.)+(?:com|jp|net|org|co\.jp|io|ai|dev|app|site)\b", re.I)


@dataclass
class ScreenConfig:
    """`settings.yaml` の `meeting.screen_capture`。"""

    enabled: bool = False
    """既定は off。相手の画面（見せている資料そのもの）を手元に残すので、使うかは人が決める。"""

    owners: list[str] = field(default_factory=lambda: ["zoom.us", "Zoom", "Google Chrome", "Comet"])
    """取り込む対象のアプリ。既定は会議アプリとブラウザだけ（画面全体は撮らない）。"""

    titles: list[str] = field(default_factory=lambda: [
        "Zoom", "ミーティング", "Google Meet", "Meet - ", "Meet – ", "Microsoft Teams", "Webex"])
    """会議のウィンドウだと分かる目印。

    ブラウザ（Chrome・Comet など）は、**この目印がタイトルに無ければ撮らない**
    （運用者 指摘 2026-09-17「Comet/Chrome はこの会議アシスタントを出していたり、
    聞きながら調べ物をしている」）。以前は「大きい窓を選ぶ」だけだったので、Zoom を使う会議でも
    ブラウザのほうが大きければ**調べ物の画面を撮ってしまう**作りだった。
    Zoom アプリのウィンドウは会議そのものなので、目印が無くても撮る。
    """

    exclude_titles: list[str] = field(default_factory=lambda: [
        "会議アシスタント", "会議ダッシュボード", "127.0.0.1", "localhost"])
    """この語を含むウィンドウは撮らない。自分の画面（この道具）を撮り返さないため。"""

    interval_sec: float = 6.0
    change_threshold: float = 0.02
    """前の 1 枚と違う画素がこの割合を超えたら残す（0〜1）。

    実測（2026-09-16・1600x900 の画面）: 同じページを撮り直すと 0.000、別のページだと 0.027。
    0.02 なら「同じ画面は残さず、別の画面は残す」。相手のカメラ映像が大きく映っていると毎回変わるので、
    `max_shots` で頭打ちにする。
    """

    max_shots: int = 400
    """1 会議で残す上限（40 分の共有でも足りる）。"""

    ocr: bool = True

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ScreenConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (values or {}).items() if key in known})


MEETING_APPS = {"zoom.us", "Zoom", "Microsoft Teams", "Webex", "Webex Meetings"}
"""会議そのもののアプリ。このアプリの窓は、タイトルの目印が無くても会議の窓とみなす。"""


def find_window(config: ScreenConfig, *, lister=None) -> dict | None:
    """取り込む対象のウィンドウ。会議の窓だと分かるものだけ（見つからなければ None）。

    選び方（2026-09-17 に厳しくした。それまでは「対象アプリの中でいちばん大きい窓」だった）:

    1. `owners` に挙げたアプリの窓だけを見る
    2. **ブラウザは、タイトルに会議の目印（Google Meet 等）が無ければ撮らない**
       ＝ 会議を聞きながらの調べ物・この道具自身の画面を撮らない
    3. `exclude_titles` を含む窓は撮らない（自分の画面を撮り返さない）
    4. 残ったものの中でいちばん大きい窓（マルチディスプレイでも、撮るのは**その窓だけ**。
       `screencapture -l <窓>` は窓を撮るので、どの画面に置いていても、ほかの画面は写らない）
    """
    rows = (lister or _windows)()
    found = []
    for row in rows:
        owner = str(row.get("owner", ""))
        title = str(row.get("title", ""))
        if owner not in config.owners or not title:
            continue
        if any(mark and mark in title for mark in config.exclude_titles):
            continue
        area = float(row.get("width", 0)) * float(row.get("height", 0))
        if area < 200_000:            # 小さい窓（通知・ツールバー）は対象外
            continue
        marked = any(mark and mark in title for mark in config.titles)
        if not marked and owner not in MEETING_APPS:
            continue                  # ブラウザの「会議ではない窓」は撮らない
        found.append((marked, area, row))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return found[0][2]


def meeting_key(*texts: str) -> dict | None:
    """会議の手がかり（`{"kind": "zoom", "key": "8412345678"}`）。見つからなければ None。

    これは「同じ会議か」を見分けるためだけに使う。会議の中身ではない。
    2026-09-18 時点では**記録するだけ**（まとめる仕組みはこれから。PLAN-cloud-hybrid.md の ④）。
    """
    for text in texts:
        for kind, pattern in MEETING_KEYS:
            found = pattern.search(str(text or ""))
            if found:
                return {"kind": kind, "key": re.sub(r"\s+", "", found.group(1))}
    return None


def _windows() -> list[dict]:
    """いま画面に出ているウィンドウ（Quartz）。画面収録の許可が要る（会議と同じ許可）。"""
    import Quartz

    rows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID)
    found = []
    for row in rows:
        bounds = row.get("kCGWindowBounds") or {}
        found.append({"id": int(row.get("kCGWindowNumber", 0)),
                      "owner": str(row.get("kCGWindowOwnerName", "")),
                      "title": str(row.get("kCGWindowName") or ""),
                      "width": float(bounds.get("Width", 0)), "height": float(bounds.get("Height", 0))})
    return found


def capture_window(window_id: int, target: Path, *, runner=subprocess.run) -> bool:
    """ウィンドウを 1 枚取り込む（影なし・音なし）。失敗しても例外にしない。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    result = runner(["screencapture", "-x", "-o", "-t", "jpg", "-l", str(window_id), str(target)],
                    capture_output=True, text=True, timeout=20)
    return getattr(result, "returncode", 1) == 0 and target.exists() and target.stat().st_size > 0


def fingerprint(path: Path, *, runner=subprocess.run) -> bytes:
    """見た目のあらまし（64x36 の白黒）。前の 1 枚と比べて、変わったかだけを見る。"""
    result = runner(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                     "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"],
                    capture_output=True, timeout=20)
    return result.stdout or b""


PIXEL_CHANGED = 24
"""1 画素が「変わった」とみなす明るさの差。白い画面どうしでも、文字や図が変われば差が出る。"""


def difference(before: bytes, after: bytes) -> float:
    """2 枚の違い＝**はっきり変わった画素の割合**（0〜1）。

    平均の差では駄目だった（実測 2026-09-16: 別々のページでも 0.007 にしかならない）。
    白地が大半の画面では、平均を取ると変化が薄まる。「何割の画素が変わったか」で見る。
    """
    if not before or not after or len(before) != len(after):
        return 1.0
    changed = sum(1 for left, right in zip(before, after) if abs(left - right) > PIXEL_CHANGED)
    return changed / len(before)


def read_text(image: Path, ocr_binary: Path, *, runner=subprocess.run) -> list[dict]:
    """画像の文字を読む（別プロセス）。落ちてもこの 1 枚を諦めるだけ。"""
    try:
        result = runner([str(ocr_binary), str(image)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("画面の文字を読めませんでした: %s", error)
        return []
    if getattr(result, "returncode", 1) != 0 or not result.stdout:
        logger.warning("画面の文字を読めませんでした: %s", (result.stderr or "")[-120:])
        return []
    try:
        return list(json.loads(result.stdout).get("lines") or [])
    except json.JSONDecodeError:
        return []


def pick_page(lines: list[dict]) -> dict:
    """読み取った行から、URL と「ページ名らしき行」を選ぶ。

    ページ名は**いちばん大きな文字の行**を採る（見出しが大きいという当たり前の性質を使う）。
    """
    texts = [str(line.get("text", "")).strip() for line in lines]
    urls, domains = [], []
    for text in texts:
        urls += [url.rstrip("。、）)") for url in URL_PATTERN.findall(text)]
        if not URL_PATTERN.search(text):
            domains += DOMAIN_PATTERN.findall(text)
    title = ""
    if lines:
        biggest = max(lines, key=lambda line: float(line.get("h", 0)))
        if len(str(biggest.get("text", "")).strip()) >= 4:
            title = str(biggest["text"]).strip()
    return {"urls": list(dict.fromkeys(urls)), "domains": list(dict.fromkeys(domains))[:3],
            "title": title, "text": "\n".join(texts)[:1500]}


class ScreenWatcher:
    """会議中、一定間隔で画面共有を控える。会議の邪魔をしないよう、別スレッドで静かに回す。"""

    def __init__(self, session_dir: Path, config: ScreenConfig, ocr_binary: Path | None,
                 *, clock=time.monotonic, started_at: float | None = None) -> None:
        self.session_dir = Path(session_dir)
        self.config = config
        self.ocr_binary = ocr_binary
        self._clock = clock
        self._started_at = started_at
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: bytes = b""
        self.shots = 0
        self.skipped = 0

    def start(self, meeting_clock) -> None:
        """`meeting_clock()` は会議の経過秒を返す関数（字幕の時刻と揃えるため）。"""
        if not self.config.enabled:
            return
        self._meeting_clock = meeting_clock
        self._thread = threading.Thread(target=self._loop, daemon=True, name="screen-watch")
        self._thread.start()
        logger.info("画面共有の取り込みを始めます（%.0f 秒ごと・対象: %s）",
                    self.config.interval_sec, "・".join(self.config.owners))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.config.interval_sec):
            if self.shots >= self.config.max_shots:
                continue
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — 画面が取れなくても会議は続ける
                logger.debug("画面の取り込みに失敗", exc_info=True)

    def tick(self) -> dict | None:
        """1 回ぶん。変わっていなければ何も残さない。"""
        window = find_window(self.config)
        if window is None:
            return None
        at = float(self._meeting_clock())
        temporary = self.session_dir / SCREENS_DIR / f"_tmp.jpg"
        if not capture_window(window["id"], temporary):
            return None
        mark = fingerprint(temporary)
        if difference(self._last, mark) < self.config.change_threshold:
            self.skipped += 1
            temporary.unlink(missing_ok=True)
            return None
        self._last = mark
        target = temporary.with_name(f"{int(at):06d}.jpg")
        temporary.replace(target)
        entry = {"at": round(at, 1), "file": f"{SCREENS_DIR}/{target.name}",
                 "window": window.get("title", ""), "owner": window.get("owner", "")}
        if self.config.ocr and self.ocr_binary is not None:
            entry.update(pick_page(read_text(target, self.ocr_binary)))
        with (self.session_dir / SCREENS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._remember_meeting(entry)
        self.shots += 1
        return entry

    def _remember_meeting(self, entry: dict) -> None:
        """どの会議だったかの手がかりを 1 度だけ残す（同じ会議を 1 つにまとめる下ごしらえ）。"""
        path = self.session_dir / MEETING_KEY_FILE
        if path.exists():
            return
        found = meeting_key(entry.get("window", ""), *(entry.get("urls") or []), entry.get("title", ""))
        if not found:
            return
        path.write_text(json.dumps({**found, "at": entry.get("at", 0),
                                    "window": entry.get("window", "")}, ensure_ascii=False),
                        encoding="utf-8")
        logger.info("会議の手がかり: %s %s", found["kind"], found["key"])


def build_ocr(repo: Path) -> Path | None:
    """文字を読む小さな道具を用意する（初回だけ 5 秒ほどでビルド）。"""
    source = repo / "src" / "screen" / "ocr.swift"
    target = repo / "workspace" / "bin" / "ocr"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(["swiftc", "-O", str(source), "-o", str(target)],
                                capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("文字を読む道具を用意できませんでした（画面の文字は読みません）: %s", error)
        return None
    if result.returncode != 0:
        logger.warning("文字を読む道具のビルドに失敗しました: %s", result.stderr[-200:])
        return None
    return target


MEETING_HOSTS = ("meet.google.com", "zoom.us", "teams.microsoft.com", "teams.live.com",
                 "webex.com", "whereby.com")
"""会議アプリ自身の住所。「見せられたページ」には数えない。

2026-09-18 運用者 指摘「画面共有で見せられたページが、GoogleMeet だと画面の URL で意味がない」。
実会議の記録を見ると、どの控えにも `https://meet.google.com/abc-defg-hij?pli=1` だけが入っていた。
ブラウザのアドレス欄を読んでいるので当然で、**共有されている中身とは関係ない**。

ただし取り除くのは「見せられたページ」の一覧だけ。どの会議だったかの手がかり
（`meeting_key`）には要るので、控えそのものからは消さない。
"""


def is_meeting_url(url: str) -> bool:
    """会議アプリ自身の URL か。中身ではなく入れ物なので、ページとしては数えない。"""
    lowered = str(url or "").lower()
    return any(host in lowered for host in MEETING_HOSTS)


def shown_urls(entry: dict) -> list[str]:
    """その控えで実際に「見せられていた」URL（会議アプリ自身は除く）。"""
    return [url for url in (entry.get("urls") or []) if not is_meeting_url(url)]


def pages(session_dir: Path) -> list[dict]:
    """その会議で見せられたページ（同じ URL はまとめる）。議事録の材料と画面に出す。"""
    path = Path(session_dir) / SCREENS_FILE
    if not path.exists():
        return []
    found: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        urls = shown_urls(entry)
        key = (urls or [entry.get("title", "")])[0] or entry.get("file", "")
        current = found.setdefault(key, {"url": (urls or [""])[0], "title": entry.get("title", ""),
                                         "first_at": entry.get("at", 0), "shots": 0, "file": entry.get("file", "")})
        current["shots"] += 1
        if not current["title"] and entry.get("title"):
            current["title"] = entry["title"]
    return sorted(found.values(), key=lambda entry: entry["first_at"])
