"""ScreenCaptureKit でシステム音声を取得する層。"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import CoreMedia
import Quartz
import ScreenCaptureKit
import libdispatch
import numpy as np
import objc
from Foundation import NSObject

logger = logging.getLogger(__name__)


@dataclass
class SckConfig:
    """ScreenCaptureKit の音声取得設定。"""

    sample_rate: int = 48000
    channels: int = 1
    exclude_current_process: bool = True
    frame_sec: float = 0.1
    app_bundle_ids: list[str] = field(default_factory=list)
    """空なら画面全体（全アプリ）の音。指定するとその bundle id のアプリの音だけを拾う
    （例: ["us.zoom.xos"]。通知音や他アプリの動画音声を議事録に入れないための絞り込み。§6-5）。"""


class _Reblocker:
    """PTS を保ちながら任意長の音声を固定長フレームへ切り直す。"""

    def __init__(self, sample_rate: int, frame_sec: float) -> None:
        self.sample_rate = sample_rate
        self.frame_sec = frame_sec
        self.frame_samples = int(round(sample_rate * frame_sec))
        if self.frame_samples <= 0:
            raise ValueError("frame_sec は 1 サンプル以上になる値にしてください。")
        self._pending = np.empty(0, dtype=np.float32)
        self._last_end_pts: float | None = None
        self.gap_total_sec = 0.0
        self.gap_max_sec = 0.0
        self.frames_out = 0

    def push(self, samples: np.ndarray, pts_sec: float) -> list[np.ndarray]:
        """サンプルを追加し、完成した固定長フレームを返す。"""
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._last_end_pts is not None:
            gap_sec = pts_sec - self._last_end_pts
            if gap_sec > self.frame_sec:
                gap_samples = int(round(gap_sec * self.sample_rate))
                self._pending = np.concatenate((self._pending, np.zeros(gap_samples, dtype=np.float32)))
                self.gap_total_sec += gap_samples / self.sample_rate
                self.gap_max_sec = max(self.gap_max_sec, gap_samples / self.sample_rate)
        self._pending = np.concatenate((self._pending, mono))
        self._last_end_pts = pts_sec + len(mono) / self.sample_rate
        frames: list[np.ndarray] = []
        while len(self._pending) >= self.frame_samples:
            frames.append(self._pending[:self.frame_samples].copy())
            self._pending = self._pending[self.frame_samples:]
            self.frames_out += 1
        return frames

    def flush(self) -> np.ndarray | None:
        """未完了の端数を返し、内部の端数を空にする。"""
        if not len(self._pending):
            return None
        remaining = self._pending.copy()
        self._pending = np.empty(0, dtype=np.float32)
        return remaining


def has_screen_capture_permission() -> bool:
    """画面収録の事前許可があるかを返す。"""
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def request_screen_capture_permission() -> bool:
    """画面収録の許可ダイアログを要求し、要求結果を返す。"""
    return bool(Quartz.CGRequestScreenCaptureAccess())


def responsible_app_name() -> str:
    """画面収録の許可先になる、いちばん外側の macOS アプリケーション名を返す。

    TCC の責任アプリは**最も外側**の .app（Agent / VS Code / Terminal）。venv の python 自体が
    `Python.app/Contents/MacOS/Python` の下にあるので、最初に見つかった .app を返すと "Python" に
    なってしまう（bb 実測 2026-09-09）。root まで遡って最後に見つかった .app を採る。
    """
    pid = str(os.getpid())
    fallback = "python"
    outermost: str | None = None
    is_current_process = True
    while pid and pid != "0":
        result = subprocess.run(
            ["ps", "-o", "ppid=,comm=", "-p", pid],
            check=False,
            capture_output=True,
            text=True,
        )
        fields = result.stdout.strip().split(maxsplit=1)
        if len(fields) != 2:
            break
        parent_pid, command = fields
        if is_current_process:
            fallback = command.rsplit("/", maxsplit=1)[-1] or fallback
            is_current_process = False
        app_marker = ".app/"
        if app_marker in command:
            app_path = command.split(app_marker, maxsplit=1)[0]
            outermost = app_path.rsplit("/", maxsplit=1)[-1]
        pid = parent_pid
    return outermost or fallback


def python_app_path(base_prefix: str | None = None) -> str | None:
    """許可先が "Python" のとき、システム設定の「＋」で選ぶ Python.app の場所を返す。

    なぜ要るか（2026-09-24・本番で発生）: `.app` から起こすと許可先は Homebrew の Python.app
      （`…/Python.framework/Versions/3.x/Resources/Python.app`）になる。この項目が一覧から消えると、
      「『Python』を許可してください」と言われても**どこを足せばよいか分からない**。
    """
    app = os.path.join(base_prefix or sys.base_prefix, "Resources", "Python.app")
    return app if os.path.isdir(app) else None


def permission_help(app: str, request=None, app_path=None) -> str:
    """許可が無いときの案内文。ついでに許可を要求し、一覧に項目を載せる。

    `CGRequestScreenCaptureAccess` は、一覧に無ければ項目を足し（オフの状態で）、
    初回ならダイアログも出す。項目が載れば、あとはスイッチを入れるだけになる。
    """
    try:
        (request or request_screen_capture_permission)()
    except Exception:  # noqa: BLE001 - 案内を出すことが優先
        pass
    lines = [f"システム設定 > プライバシーとセキュリティ > 画面収録とシステムオーディオ録音 で『{app}』を許可してください。"]
    if app.lower() == "python":
        path = (app_path or python_app_path)()
        if path:
            lines.append(f"  一覧に Python が無ければ「＋」で次を選ぶ（Cmd+Shift+G で貼り付け）: {path}")
    lines.append("  許可したら、会議アシスタント（またはターミナル）を終了して起動し直してください。")
    return "\n".join(lines)


class _StreamDelegate(NSObject):
    """PyObjC から SCK の音声コールバックを受けるデリゲート。"""

    __pyobjc_protocols__ = [objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]

    def initWithOwner_(self, owner: "SckSystemAudio") -> "_StreamDelegate":
        self = objc.super(_StreamDelegate, self).init()
        if self is None:
            return None
        self.owner = owner
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream: object, sample_buffer: object, output_type: int) -> None:
        if output_type != ScreenCaptureKit.SCStreamOutputTypeAudio:
            return
        self.owner._handle_sample_buffer(sample_buffer)

    def stream_didStopWithError_(self, stream: object, error: object) -> None:
        self.owner._stream_error = RuntimeError(str(error))
        logger.error("ScreenCaptureKit stream stopped: %s", error)


class SckSystemAudio:
    """ScreenCaptureKit のシステム音声をモノラル固定長フレームで渡す。"""

    def __init__(self, cfg: SckConfig, on_frame: Callable[[np.ndarray], None]) -> None:
        self.cfg = cfg
        self.on_frame = on_frame
        self.first_frame_at: float | None = None
        self._reblocker = _Reblocker(cfg.sample_rate, cfg.frame_sec)
        self._stream: object | None = None
        self._delegate: _StreamDelegate | None = None
        self._queue: object | None = None
        self._stream_error: RuntimeError | None = None
        self._started = False
        self._buffer_pts: list[float] = []
        self._buffer_samples: list[int] = []
        self._received_buffers = 0
        self._last_pts: float | None = None
        self._first_pts: float | None = None
        self._last_samples = 0
        self._format_flags: int | None = None
        self.queue_mode = "none"

    def start(self, timeout: float = 5.0) -> None:
        """SCK ストリームを開始し、初期化失敗を呼び出し元へ伝える。"""
        if not has_screen_capture_permission():
            raise PermissionError("画面収録の許可がありません。")
        content_event = threading.Event()
        content_box: dict[str, object] = {}

        def _content_ready(content: object, error: object) -> None:
            content_box["content"] = content
            content_box["error"] = error
            content_event.set()

        ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_(_content_ready)
        if not content_event.wait(timeout):
            raise RuntimeError("共有コンテンツの取得がタイムアウトしました。")
        if content_box.get("error") is not None or content_box.get("content") is None:
            raise RuntimeError(f"共有コンテンツを取得できません: {content_box.get('error')}")
        displays = content_box["content"].displays()
        if not displays:
            raise RuntimeError("共有可能なディスプレイがありません。")
        content_filter = self._make_filter(content_box["content"], displays[0])
        configuration = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
        configuration.setCapturesAudio_(True)
        configuration.setSampleRate_(self.cfg.sample_rate)
        configuration.setChannelCount_(self.cfg.channels)
        configuration.setExcludesCurrentProcessAudio_(self.cfg.exclude_current_process)
        configuration.setWidth_(2)
        configuration.setHeight_(2)
        configuration.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 1))
        self._delegate = _StreamDelegate.alloc().initWithOwner_(self)
        self._stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(content_filter, configuration, self._delegate)
        try:
            self._stream.addStreamOutput_type_sampleHandlerQueue_error_(self._delegate, ScreenCaptureKit.SCStreamOutputTypeAudio, None, None)
        except Exception:
            self._queue = libdispatch.dispatch_queue_create(b"sck-audio", None)
            self._stream.addStreamOutput_type_sampleHandlerQueue_error_(self._delegate, ScreenCaptureKit.SCStreamOutputTypeAudio, self._queue, None)
            self.queue_mode = "libdispatch"
        started = threading.Event()
        start_error: dict[str, object] = {}

        def _started(error: object) -> None:
            start_error["error"] = error
            started.set()

        self._stream.startCaptureWithCompletionHandler_(_started)
        if not started.wait(timeout) or start_error.get("error") is not None:
            raise RuntimeError(f"ScreenCaptureKit を開始できません: {start_error.get('error')}")
        self._started = True
        logger.info("ScreenCaptureKit audio capture started (queue=%s)", self.queue_mode)

    def _make_filter(self, content: object, display: object) -> object:
        """アプリ絞り込みの有無で SCContentFilter を作る。"""
        if not self.cfg.app_bundle_ids:
            return ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        wanted = set(self.cfg.app_bundle_ids)
        applications = [app for app in content.applications() if str(app.bundleIdentifier()) in wanted]
        if not applications:
            raise RuntimeError(f"指定した bundle id のアプリが起動していません: {sorted(wanted)}")
        logger.info("ScreenCaptureKit をアプリで絞り込み: %s", [str(app.bundleIdentifier()) for app in applications])
        return ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_includingApplications_exceptingWindows_(
            display, applications, []
        )

    def stop(self) -> None:
        """SCK ストリームを停止する。"""
        if not self._stream or not self._started:
            return
        stopped = threading.Event()

        def _stopped(error: object) -> None:
            if error is not None:
                logger.warning("ScreenCaptureKit の停止時にエラー: %s", error)
            stopped.set()

        self._stream.stopCaptureWithCompletionHandler_(_stopped)
        stopped.wait(5.0)
        self._started = False
        logger.info("ScreenCaptureKit audio capture stopped")

    def _handle_sample_buffer(self, sample_buffer: object) -> None:
        """SCK の音声バッファを float32 に変換して再ブロックする。"""
        try:
            block = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
            length = CoreMedia.CMBlockBufferGetDataLength(block)
            status, data = CoreMedia.CMBlockBufferCopyDataBytes(block, 0, length, None)
            if status != 0:
                raise RuntimeError(f"CMBlockBufferCopyDataBytes failed: {status}")
            pts = CoreMedia.CMSampleBufferGetPresentationTimeStamp(sample_buffer)
            pts_sec = pts.value / pts.timescale
            samples = np.frombuffer(data, dtype=np.float32).copy()
            sample_count = int(CoreMedia.CMSampleBufferGetNumSamples(sample_buffer))
            if sample_count != len(samples):
                logger.warning("SCK の sample 数が float32 要素数と異なります: %d != %d", sample_count, len(samples))
            self._received_buffers += 1
            self._buffer_pts.append(pts_sec)
            self._buffer_samples.append(sample_count)
            self._last_pts = pts_sec
            self._last_samples = len(samples)
            if self._first_pts is None:
                self._first_pts = pts_sec
            if self.first_frame_at is None:
                self.first_frame_at = time.time()
            self._record_format_flags(sample_buffer)
            for frame in self._reblocker.push(samples, pts_sec):
                self.on_frame(frame)
        except Exception:
            logger.exception("ScreenCaptureKit 音声コールバックの処理に失敗しました")

    def _record_format_flags(self, sample_buffer: object) -> None:
        """取得できる環境では音声フォーマットフラグを記録する。"""
        if self._format_flags is not None:
            return
        try:
            description = CoreMedia.CMSampleBufferGetFormatDescription(sample_buffer)
            basic = CoreMedia.CMAudioFormatDescriptionGetStreamBasicDescription(description)
            self._format_flags = int(basic.mFormatFlags)
        except Exception:
            logger.debug("SCK 音声フォーマットフラグを取得できませんでした", exc_info=True)

    def stats(self) -> dict[str, float | int | None | str | list[float]]:
        """診断用の受信・PTS・再ブロック統計を返す。"""
        # PTS はホスト時計の絶対値なので、最初の PTS からの経過で比べる（bb レビューで修正）。
        #   受信した音の長さ = (最終 PTS + 最終バッファ長) − 最初の PTS。出力 = フレーム化済み＋端数。
        output_sec = self._reblocker.frames_out * self.cfg.frame_sec + len(self._reblocker._pending) / self.cfg.sample_rate
        received_sec = (
            None
            if self._last_pts is None or self._first_pts is None
            else (self._last_pts + self._last_samples / self.cfg.sample_rate) - self._first_pts
        )
        drift = None if received_sec is None else received_sec - output_sec
        return {
            "gap_total_sec": self._reblocker.gap_total_sec,
            "gap_max_sec": self._reblocker.gap_max_sec,
            "frames_out": self._reblocker.frames_out,
            "buffers": self._received_buffers,
            "last_pts": self._last_pts,
            "received_sec": received_sec,
            "output_sec": output_sec,
            "pts_vs_samples_drift_sec": drift,
            "format_flags": self._format_flags,
            "queue_mode": self.queue_mode,
            "buffer_pts": list(self._buffer_pts),
            "buffer_samples": list(self._buffer_samples),
        }
