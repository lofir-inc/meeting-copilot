"""会議（ループバック）モードのオーケストレーター。

対面インタビュー用の `orchestrator.py` とは**独立したクラス**。あちらには1行も触れない。
要約サイクル周りが一部重複するが、実会議で運用が固まるまでは分けておく方が安い。

    [自分のマイク]   --stream self---> VAD --> 話者= 固定（判定しない）--\\
                                                                        >-- STT --> transcript --> LLM
    [BlackHole 2ch]  --stream remote-> VAD --> EnrolledDiarizer -------/
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np

from src.audio import devices as dev
from src.audio.enrolled_diarizer import DiarizerConfig, EnrolledDiarizer
from src.audio.multi_capture import MultiAudioConfig, MultiCapture, SourceConfig
from src.audio.sck_capture import has_screen_capture_permission, permission_help, responsible_app_name
from src.audio.vad import AudioChunk, ChunkBuilder, VadConfig
from src.audio.turn_splitter import TurnSplitConfig, slice_chunk, split_turns
from src.audio.segment_labeler import label_segments
from src import clients, finish
from src.screen.capture import ScreenConfig, ScreenWatcher, build_ocr
from src.audio.voice_library import (
    VoiceLibrary,
    VoiceLibraryConfig,
    read_wav_mono,
    record_match,
    voices_from_session,
)
from src.enrollment import CaptureFrameSource, EnrollmentConfig, failed_enrollment_lines, run_enrollment
from src.stt.external_consent import (
    ExternalSendRefused,
    ExternalSttConfig,
    approve,
    LOGIN_EXPIRED,
    login_and_wait,
    audio_minutes,
    billing_enabled,
    record_send,
)
from src.stt.gemini_transcribe import TranscribeApi, TranscribeConfig, load_key
from src.stt.live_batch import GAPS_FILE, AudioWindow, LiveBatchConfig, LiveBatchStt
from src.stt.live_batch import config_from as live_config_from
from src.llm.ollama_client import LlmConfig, OllamaClient
from src.dispatch import DispatchConfig, DispatchResult, dispatch, finish_command, write_minutes_handoff
from src.llm.claude_cli_client import ClaudeCliClient, final_pass_prompt
from src.llm.gemini_client import GeminiClient, GeminiLlmConfig
from src.llm.factcheck import Claim, FactCheckConfig, FactCheckLoop, FactChecker
from src.llm.meeting_state import MeetingState, apply_delta
from src.llm.state_updater import StateLoop, StateUpdater, StateUpdaterConfig, UpdateStats
from src.task_hub import TaskHubBridge, TaskHubConfig, TaskHubUnavailable
from src.agent_client import AgentClient
from src.output.markdown_writer import MarkdownWriter, OutputConfig
from src.stt.whisper_client import SttConfig, TranscriptSegment, WhisperClient
from src.system_load import wait_until_quiet
from src.meeting.prep import (  # noqa: F401 — 定数は外（テスト・道具）からも参照される
    PREP_DIGEST_LOCAL_INPUT,
    PREP_DIGEST_MAX_INPUT,
    PREP_MAX_BYTES,
    PREP_MAX_CHARS,
    PREP_SUFFIXES,
    PrepMaterials,
)
from src.meeting.verify import (  # noqa: F401
    VERIFY_CONTEXT_SEC,
    VERIFY_FILE,
    VERIFY_ENGINES,
    VERIFY_TIMEOUT_SEC,
    GeminiVerifyClient,
    Verifier,
)
from src.text.glossary import apply_glossary, load_glossary, load_protected
from src.text.stitch import stitch
from src.text.terms import extract_terms, prompt_line, save_terms
from src.text import task_hub_dictionary
from src.ui.bus import EventBus
from src.ui.server import PortInUse, UiConfig, build_app, start_ui_server

logger = logging.getLogger(__name__)

SELF_KEY = "self"
REMOTE_KEY = "remote"

EXTERNAL_CHECK_TIMEOUT_SEC = 20.0
"""会議前に課金の確認を試すときの待ち時間。起動を長く止めない。"""

RENAMES_FILE = "renames.jsonl"
"""画面でまとめて付けた話者名（旧名 → 新名）。会議のあとの作り直しで当て直すのに使う。

2026-09-14 の会議で必要になった: 画面で「不明話者1 → 実名」と付けたのに、その名前が
声の主（声紋）に結び付いていなかったため、**作り直すと名前が消えた**（736 行）。
"""

# 事前資料（PREP_*）と裏取り（VERIFY_*）の決まりごとは `src/meeting/` の部品が持つ。
#   ここで読み込み直しているのは、外（テスト・道具）からこの名前で参照されているため。


@dataclass
class MeetingConfig:
    """会議モードの設定（`settings.yaml` の `meeting:` セクション）。"""

    self_name: str = "自分"
    """自分の発話に付ける固定ラベル。マイクを物理的に分けているので判定しない。"""

    mic_candidates: list[str] = field(default_factory=list)
    """自分のマイクの候補名。**優先順**に並べる。日によって変わるので番号は書かない。"""

    listening_device_for: dict[str, str] = field(default_factory=dict)
    """マイク名 → その日に「聴く側」になるデバイス名。

    束に入っている必要があるのは**聴く側**であって、マイクではない。
    据え置きの Shure MV7 で喋って有線イヤホン（USB Audio Device）で聴く運用があるため、
    「今日のマイクが束に入っているか」では可否を判定できない。
    ヘッドセット類はマイクと出力が同名なので、書かなければマイク名がそのまま使われる。
    """

    loopback_candidates: list[str] = field(default_factory=lambda: ["BlackHole 2ch"])
    """相手側の声を拾うループバックデバイスの候補名。"""

    capture_backend: str = "screencapturekit"
    """相手音声の取得方法。`screencapturekit` または `blackhole` を指定する。"""
    sck: dict = field(default_factory=dict)
    """ScreenCaptureKit 設定の上書き。"""
    beep_selftest: bool = True
    """起動時にビープの自己診断を実行するか。"""
    beep_hz: int = 880
    """自己診断用ビープの周波数。"""
    beep_sec: float = 0.2
    """自己診断用ビープの長さ。"""
    beep_dbfs: float = -20.0
    """自己診断用ビープの音量。"""

    multi_output_name: str = "会議録音用（複数出力装置）"
    """会議アプリの出力先に指定すべき複数出力装置の名前。点検にだけ使う。"""

    app_name: str = "会議アプリ"
    """案内文に出す会議アプリ名（Zoom / Google Meet / Teams …）。

    ループバック方式は**システムの音声出力を丸ごと拾う**ので、アプリには依存しない。
    Zoom でも Meet でも Teams でも同じように動く。この設定は案内文の文言だけに使う。
    """

    similarity_threshold: float = 0.75
    max_adapt_embeddings: int = 10
    summary_mode: str = "rolling"
    """要約方式。`rolling` は状態差分方式、`legacy` は従来の3分要約を使う。"""

    state: dict = field(default_factory=dict)
    """ローリング状態更新設定の上書き。"""

    factcheck: dict = field(default_factory=dict)
    """ファクトチェック設定の上書き（`FactCheckConfig`）。既定は無効。"""

    vad: dict = field(default_factory=dict)
    """会議専用の VAD 設定上書き。"""

    turn_split: dict = field(default_factory=dict)
    """会議専用の話者ターン分割設定上書き。"""

    diarizer: dict = field(default_factory=dict)
    """会議専用の話者照合設定上書き。"""
    enroll_duration_sec: float = 5.0
    level_check_sec: float = 3.0
    first_frame_timeout_sec: float = 5.0
    """両ストリームの最初のフレームが揃うまで待つ上限。超えたら警告して先へ進む。"""
    skip_enrollment: bool = False
    """True なら登録フローを飛ばす（既存 speakers.json をそのまま使う／全員 不明話者N）。"""
    rename_anchor_rows: int = 5
    """画面で名前を付けたとき、その名前の行から何件の声を錨にするか。

    多すぎると取り違えた行を巻き込む。長いほうから数件だけ使う。
    """

    anchor_min_sec: float = 1.5
    """UI で行を直したとき、その声を話者として覚える最小の長さ。これより短い行はラベルだけ変える。"""
    max_row_voices: int = 5000
    """行ごとの声（embedding）を覚えておく件数の上限。古いものから捨てる。"""
    open_folder_after: bool = True
    """終了後にセッションのフォルダを Finder で開くか（パスを辿らなくていいように）。"""
    finalize_after: bool = True
    """会議のあと、録音から全文を起こし直して議事録の材料を作り直すか（`scripts/finalize_meeting.py`）。"""
    wait_for_apps: list[str] = field(default_factory=list)  # 2026-09-15: 既定で待たない（MacWhisper 非常駐）
    """会議後の作り直しを始める前に、処理が落ち着くのを待つアプリ名（空なら待たない）。"""
    wait_cpu_percent: float = 20.0
    """このアプリの CPU 合計がこれを下回ったら「落ち着いた」とみなす。"""
    wait_timeout_sec: float = 1800.0
    """待つ上限。超えたら待たずに始める（待ち続けて何も出ないより動くほうがよい）。"""
    silence_alert_sec: float = 120.0
    """本編に入ってからこの秒数まったく音が来なければ画面で警告する（0 で無効）。"""
    keep_backlog_sec: float = 180.0
    """本編に入るとき、直前の何秒ぶんを文字起こしに回すか。

    0 にすると本編前の音は全部捨てる（従来の挙動）。既定で 3 分残すのは、
    参加 → 挨拶しながら相手側のレベルを測る → その挨拶の行で話者に名前を付ける
    という運用（運用者 2026-09-12）を成立させるため。
    """
    review_lead_min: float = 10.0
    """予定時間を入れたとき、終了の何分前に全文を見直すか（画面の「予定」欄から使う）。"""
    voice_library: dict = field(default_factory=dict)
    """声の台帳（会議をまたいで「前にも出た人」を候補に出す）。中身は `src/audio/voice_library.py`。

    自動では名前を付けず、候補を出して人が押す（会議をまたぐと本人と別人の類似度が重なる・実測）。
    """

    screen_capture: dict = field(default_factory=dict)
    """画面共有の取り込み（何のページの話かを残す）。中身は `src/screen/capture.py`。

    既定は off。相手が見せている資料そのものを手元に残すので、使うかは人が決める。
    """

    finish_mode: str = "ask"
    """会議の終わりに、仕上げ（作り直し・議事録・辞書の候補・声の台帳）をどうするか。

    ask=画面で聞いてから走らせる（既定）／auto=そのまま走らせる／later=いつも後回し。
    2026-09-16 運用者 指示: 対面の商談で終わったらすぐ移動する・Wi-Fi が切れる場面がある。
    「数分かかる処理が勝手に始まって、途中で落ちる」を避ける。録音と全文はここまでで残っている。
    """

    finish_wait_sec: float = 600.0
    """仕上げるかの返事を待つ上限。超えたら**後回し**に倒す（勝手に始めない）。"""

    open_library_after: bool = True
    """会議のあとに辞書の候補があれば、画面「会議アシスタント」を開く（`scripts/library_ui.py`）。"""

    verify_engine: str = "auto"
    """🔍（画面から頼む裏取り）の手段。auto | claude_cli | gemini | none。

    auto は ①Claude CLI（サブスクがあれば・従量なし）→ ②Gemini＋Google 検索（**その会議で外への
    テキスト送信を承認したとき**だけ）→ ③なし。中身は `src/meeting/verify.py`。
    """

    prep_reset: bool = True
    """会議ごとに事前資料を空から始めるか。既定 true（運用者 依頼 2026-09-14）。

    false にすると、従来どおり `workspace/prep/` の中身をセッションへコピーする。
    共有フォルダを使い回していたせいで、09-11 の定例が 4 月の別会議の計画を評価基準にしていた。
    資料は画面（ダッシュボードの「事前資料」）から、会議の途中でも足せる。
    """

    glossary_path: str = "config/glossary.yaml"
    """用語辞書のパス。文字起こしの結果を決まった表記へ置き換える（Whisper のヒントには使わない）。"""
    aizuchi_mark: bool = True
    """相づちだけの行に印を付けるか（落とさずに残す。運用者 決定 2026-09-13）。

    議事録を作る側が読み飛ばせるようにするための印。消しはしない — 相づちは「聞いていた」
    「同意した」の記録でもあり、後から戻せなくなる。
    """
    external_stt: dict = field(default_factory=dict)
    """会議後の作り直しを外（Gemini）の文字起こしで行う設定。既定は無効。

    中身は `src.stt.external_consent.ExternalSttConfig` が読む。会議中は使わない
    （外へ出す承認を取る余地がなく、Gemini はファイル一括でリアルタイムに向かないため）。
    """
    min_avg_logprob: float | None = -1.0
    """Whisper の出力をこれより自信が低ければ捨てる。

    2026-09-11 本番を MacWhisper と突き合わせて決めた: 物音を文字にした行（「impressive」「лад振緻」等）の
    35% が -1.0 未満、MacWhisper と 7 割以上一致した行は 120 行中 3 行（うち 2 行は文字化け）だけが -1.0 未満。
    -0.8 だと物音 56% を落とせるが、正しい行も 9% 落ちる。
    """


@dataclass(eq=False)  # embedding（numpy 配列）を持つので同一性で比べる
class _RowVoice:
    """相手側の 1 行の声。"""

    start_time: float
    end_time: float
    embedding: np.ndarray
    speaker: str
    """話者判定が付けたラベル（改名前）。表示中のラベルは `_current_label` で引く。"""
    manual: bool = False
    """人が直した行。自動の付け直しでは触らない。"""


EXTERNAL_STT_ENGINES = frozenset({"gemini", "deepgram", "external"})
"""`stt.engine` が外で回す指示かどうか。

2026-09-18 に見つけた穴: ここが `== "gemini"` だけだったため、プリセット「バランス」
「速さ優先」（`stt.engine: deepgram` を入れる）を当てると、**外へ出す設定なのに
黙って手元へ落ちていた**。画面にも記録にも何も出ない。
`provider` が送り先を決めるので、ここは「外かどうか」だけを見る。
"""


class MeetingOrchestrator:
    """会議モードのメインオーケストレーター。"""

    def __init__(self, config: dict, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self._raw_config = config
        """起動時の設定そのもの（画面「会議アシスタント」が辞書の場所などを読む）。"""
        self.meeting = MeetingConfig(**config.get("meeting", {}))

        vad_values = dict(config.get("vad", {}))
        vad_values.update(self.meeting.vad)
        self._vad_cfg = VadConfig(**vad_values)
        self._turn_cfg = TurnSplitConfig(**self.meeting.turn_split)
        diarizer_values = dict(self.meeting.diarizer)
        diarizer_values.setdefault("similarity_threshold", self.meeting.similarity_threshold)
        diarizer_values.setdefault("max_adapt_embeddings", self.meeting.max_adapt_embeddings)
        self._diarizer_cfg = DiarizerConfig(**diarizer_values)
        self._audio_cfg = MultiAudioConfig(
            sample_rate=config.get("audio", {}).get("sample_rate", 48000),
            dtype=config.get("audio", {}).get("dtype", "float32"),
            chunk_duration_sec=config.get("audio", {}).get("chunk_duration_sec", 0.1),
        )

        output_cfg = OutputConfig(**config.get("output", {}))
        output_cfg.title = "会議"
        output_cfg.items_label = "前回タスク 消化状況"

        self._summary_interval = config.get("orchestrator", {}).get("summary_interval_sec", 300)

        self.whisper = WhisperClient(SttConfig(**config.get("stt", {})))
        if self.whisper.config.min_avg_logprob is None:
            self.whisper.config.min_avg_logprob = self.meeting.min_avg_logprob
        self._stt_engine = (self.whisper.config.engine or "local").lower()
        """会議**中**の文字起こしをどこで回すか。`gemini` なら 30 秒刻みで外へ投げる（方式④）。
        関門を通らなければ `local` に戻る。決まるのは `_start_live_stt()`。"""
        self._external_cfg = ExternalSttConfig.from_mapping(self.meeting.external_stt)
        self._live_stt: LiveBatchStt | None = None
        self._live_stt_label: dict[str, str] = {}
        """画面の「モデル」欄に出す、会議中の文字起こしの送り先（`_start_live_stt` で決まる）。"""
        llm_values = dict(config.get("llm", {}))
        self._final_pass = str(llm_values.pop("final_pass", "none"))
        self._llm_engine = str(llm_values.get("engine", "local")).lower()
        """会議中の要約をどこで回すか。`gemini` なら外で回す（送るのはテキストだけ・音声は送らない）。
        決まるのは `_start_external_engines()`。承認が取れなければ `local` に戻る。"""
        self._llm_gemini = dict(llm_values.get("gemini") or {})
        # 入口はいつもローカル。外へ切り替えるのは、会議ごとの承認が通ってから
        self.llm = OllamaClient(LlmConfig(**llm_values))
        self._state_cfg = StateUpdaterConfig(**self.meeting.state)
        self.writer = MarkdownWriter(output_cfg)

        # --- 起動時に確定するもの ---
        self.mic: dev.ResolvedDevice | None = None
        self.loopback: dev.ResolvedDevice | None = None
        self.capture: MultiCapture | None = None
        self.diarizer: EnrolledDiarizer | None = None
        self._builders: dict[str, ChunkBuilder] = {}

        # --- 状態 ---
        self._transcript_buffer: list[TranscriptSegment] = []
        self._buffer_lock = threading.Lock()
        self._emit_lock = threading.Lock()
        """文字起こしの出口。会議中に外で回すと 2 系統が同時に返るので、1 本ずつ通す
        （ファイルへの追記・話者ごとの集計・声紋の更新がぶつかる）。"""
        self._overall_summary: str = ""
        self._prep = PrepMaterials(
            self.session_dir,
            Path(__file__).resolve().parent.parent / "prompts",
            limit=lambda: self._state_cfg.prior_tasks_chars,
            llm=lambda: self.llm,
            # 設定（llm.engine）ではなく、**いま実際に使っている口**で決める。承認前は設定が gemini でも口は Ollama
            engine=lambda: "gemini" if isinstance(self.llm, GeminiClient) else "local",
            on_change=self._on_prep_changed,
            publish=lambda name, payload: self.bus.publish(name, payload),
        )
        """事前資料（読み込み・追加・削除・まとめ）。中身は `src/meeting/prep.py`。"""
        self._system_prompt: str = ""
        self._state_loop: StateLoop | None = None
        self._factcheck_cfg = FactCheckConfig(**{k: v for k, v in self.meeting.factcheck.items() if k in FactCheckConfig.__dataclass_fields__})
        self._factcheck_loop: FactCheckLoop | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)  # Metal GPU は同時アクセス不可
        self.bus = EventBus()
        self._ui_cfg = UiConfig(**config.get("ui", {}))
        self._ui_port = self._ui_cfg.port
        """実際に画面を開いたポート（塞がっていたら隣へ逃げるので、設定の値とは限らない）。"""
        self._ui_thread: threading.Thread | None = None
        self._stt_latencies: deque[float] = deque(maxlen=50)
        self._last_level_at = 0.0
        self._last_metrics_at = 0.0
        self._last_drift_log_at = 0.0
        self._last_metrics_log_at = 0.0
        self._review_lock = threading.Lock()
        self._finish_requested = threading.Event()
        self._preflight_decided = threading.Event()
        self._last_sound_at: dict[str, float] = {}
        self._silence_alerted: set[str] = set()
        self.startup_warnings: list[str] = []
        """起動前に分かっている警告（古い事前資料など）。セルフチェックの画面に一緒に出す。"""
        self._preflight_answer: str | None = None
        self._external_answer: str | None = None
        self._external_decided = threading.Event()
        """画面の「会議を終了」から立てる。主ループがこれを見て終了処理へ入る。"""
        self._planned_minutes: float | None = None
        self._auto_review_at: float | None = None
        self._auto_review_done = False
        glossary_path = Path(self.meeting.glossary_path)
        if not glossary_path.is_absolute():
            glossary_path = Path(__file__).resolve().parent.parent / glossary_path
        self._clients = clients.load_cache(Path(__file__).resolve().parent.parent / clients.CACHE_FILE)
        """登録済みのクライアント一覧（手元の控え）。この会議の行き先は `clients.effective()` で決まる。"""
        self._screen_cfg = ScreenConfig.from_mapping(self.meeting.screen_capture)
        self._screens: ScreenWatcher | None = None
        """画面共有の取り込み（会議中だけ動く）。"""
        self._finish_decided = threading.Event()
        """仕上げるかの返事（画面から立てる）。"""
        self._finish_choice = ""
        self._client_decided = threading.Event()
        """会議の終わりに聞いたクライアントの返事（画面から立てる）。"""
        self._task_hub_sync = bool((config.get("task_hub") or {}).get("dictionary_sync", False))
        self._task_hub_pairs = (task_hub_dictionary.load_cache(Path(__file__).resolve().parent.parent / task_hub_dictionary.CACHE_FILE)
                             if self._task_hub_sync else [])
        """タスク管理 のクライアント辞書（手元のキャッシュ）。会議中に当てるのは名前の対だけ（`live_pairs`）。"""
        self._glossary = task_hub_dictionary.merge_pairs(load_glossary(glossary_path),
                                                      task_hub_dictionary.live_pairs(self._task_hub_pairs))
        self._glossary_protect = load_protected(glossary_path)
        self._meeting_terms: list[str] = []
        """事前資料から拾った固有名詞。この会議だけの守り札と、要約への「表記はこれ」に使う（`src/text/terms.py`）。"""
        self._stt_work: deque[tuple[float, float]] = deque(maxlen=50)
        """(処理にかかった秒, 音声の長さ秒) の直近 50 件。"""
        self._started_at = time.time()
        self._capture_started_at: float | None = None
        """録音が始まった時刻（`start_time` の 0 に当たる壁時計）。始まるまでは None。"""
        self._stt_pending = 0
        self._stt_pending_lock = threading.Lock()
        self._last_state_latency_sec: float | None = None
        self._speaker_stats: dict[str, list[int]] = {}
        self._aliases: dict[str, str] = {}
        """改名の履歴（旧名 → 新名）。改名の瞬間に文字起こし中だった発話が旧ラベルで届くので、出口で揃える。"""
        self._row_voices: dict[float, _RowVoice] = {}
        """相手側の各行の声（embedding）。行を直したときにその声を覚え、過去の「不明話者?」を付け直すのに使う。"""
        self._rows_lock = threading.Lock()
        self._task_hub_cfg = TaskHubConfig(**config.get("task_hub", {}))
        self._dispatch_cfg = DispatchConfig(**config.get("dispatch", {}))
        self._bridge: TaskHubBridge | None = None
        try:
            shared = Path(self._task_hub_cfg.shared_dir).expanduser()
            if not (shared / "task_hub_context.py").is_file():
                raise TaskHubUnavailable(f"タスク管理 の共有モジュールがありません: {shared}")
            bridge = TaskHubBridge(self._task_hub_cfg)
            self._bridge = bridge
        except TaskHubUnavailable as exc:
            logger.warning("タスク管理 連携を無効化して起動します: %s", exc)
        self._agent: AgentClient | None = None
        candidate_agent = AgentClient()
        if candidate_agent.available():
            self._agent = candidate_agent
        self._dispatched: list[DispatchResult] = []
        self._voice_cfg = VoiceLibraryConfig.from_mapping(self.meeting.voice_library)
        self._voices: VoiceLibrary | None = None
        """声の台帳。有効なときだけ読む（声紋は個人を識別できる情報なので、既定は off）。"""
        self._voice_rejected: dict[str, set[str]] = {}
        """不明話者 → 「違う」と押された候補。同じ候補を出し続けない。"""
        if self._voice_cfg.enabled:
            try:
                self._voices = VoiceLibrary(self._voice_library_path(), self._voice_cfg)
                logger.info("声の台帳を読みました: %d 人", len(self._voices))
            except (OSError, ValueError) as error:
                logger.warning("声の台帳を読めません（候補は出しません）: %s", error)
        self._verify_gemini: GeminiClient | None = None
        """🔍 を Gemini で回すときの口（会議の承認が取れたときだけ作る）。"""
        self._verify_engine = str(self.meeting.verify_engine).lower()
        if self._verify_engine not in VERIFY_ENGINES:
            logger.warning("知らない裏取りの手段です（%s）。auto として扱います", self._verify_engine)
            self._verify_engine = "auto"
        claude = ClaudeCliClient()      # サブスクの範囲で動くので従量課金は増えない
        use_claude = self._verify_engine in {"auto", "claude_cli"} and claude.available()
        self._verifier = Verifier(
            self.session_dir,
            claude if use_claude else None,
            quote_at=self._text_at,
            context_at=self._context_around,
            publish=lambda name, payload: self.bus.publish(name, payload),
        )
        """裏取り（Web 検索つき）。中身は `src/meeting/verify.py`。"""
        if not use_claude:
            self._verifier.disable(self._verify_unavailable_reason(claude.available()))

    def _voice_library_path(self) -> Path:
        path = Path(self._voice_cfg.path).expanduser()
        return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path

    # ------------------------------------------- 声の台帳（前にも出た人を候補に出す）

    def voice_suggestions(self, speaker: str) -> list[dict]:
        """不明話者に、過去の会議で覚えた人の候補を付ける（近い順）。"""
        if self._voices is None or self.diarizer is None or not speaker.startswith("不明話者") or speaker.endswith("?"):
            return []
        voice_of = getattr(self.diarizer, "voice_of", None)
        found = voice_of(speaker) if voice_of is not None else None
        if found is None or found[1] < self._voice_cfg.min_samples:
            return []
        present = set(getattr(self.diarizer, "speaker_names", []) or []) | {self.meeting.self_name}
        exclude = present | self._voice_rejected.get(speaker, set())
        return [item.to_dict() for item in self._voices.suggest(found[0], exclude=exclude)]

    def accept_voice(self, speaker: str, name: str) -> dict:
        """候補を押した: その不明話者に過去の人の名前を付ける（過去の行にも反映される）。"""
        suggestion = next((item for item in self.voice_suggestions(speaker) if item["name"] == name),
                          {"name": name, "score": None, "last_session": ""})
        self.rename_speaker(speaker, name)
        record_match(self.session_dir, speaker=speaker, suggestion=suggestion, accepted=True)
        logger.info("声の台帳から名前を付けました: %s → %s（%s）", speaker, name, suggestion.get("score"))
        return {"ok": True, "old": speaker, "new": name}

    def reject_voice(self, speaker: str, name: str) -> dict:
        """「違う」を押した: この会議では、その不明話者にその候補を出さない。"""
        suggestion = next((item for item in self.voice_suggestions(speaker) if item["name"] == name),
                          {"name": name, "score": None, "last_session": ""})
        self._voice_rejected.setdefault(speaker, set()).add(name)
        record_match(self.session_dir, speaker=speaker, suggestion=suggestion, accepted=False)
        return {"ok": True}

    def _learn_voices(self) -> None:
        """会議のあとに、名前の付いた人の声を台帳へ覚える。作り直した全文があればそちらを使う。"""
        if self._voices is None:
            return
        rows_path = self.session_dir / "transcripts_final.jsonl"
        if not rows_path.exists():
            rows_path = self.session_dir / "transcripts.jsonl"
        recording = self.session_dir / "recording_remote.wav"
        if not rows_path.exists() or not recording.exists():
            return
        try:
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:   # 会議中の改名を当てる（作り直していない全文は改名前の名前のまま）
                row["speaker"] = self._resolve_alias(str(row.get("speaker", "")))
            audio, sample_rate = read_wav_mono(recording)
            voices = voices_from_session(rows, audio, sample_rate, self_name=self.meeting.self_name,
                                         config=self._voice_cfg, embed=EnrolledDiarizer.embed)
            result = self._voices.learn(self.session_dir.name, voices)
            self._voices.save()
        except Exception:  # noqa: BLE001 — 覚えられなくても会議の記録は失われない
            logger.exception("声の台帳への記録に失敗")
            return
        if result.learned:
            print("  声を覚えました（次の会議で候補に出ます）: "
                  + "、".join(f"{name}（{count} 行）" for name, count in result.learned.items()))
        for name, reason in result.skipped.items():
            print(f"  {name} の声は覚えませんでした: {reason}")

    # ------------------------------------------- 画面「会議アシスタント」（会議中に直したらその場で効かせる）

    def _library_router(self):
        from src.ui.library import Library, build_library_router, task_hub_pusher

        repo = Path(__file__).resolve().parent.parent
        settings = lambda: self._raw_config  # noqa: E731
        library = Library(repo, settings, on_change=self._on_library_changed,
                          current_dictionary=lambda: "gemini" if self._live_stt is not None else "whisper",
                          push_to_task_hub=task_hub_pusher(repo, settings))
        return build_library_router(library)

    def _on_library_changed(self, kind: str) -> None:
        """画面で辞書や声の台帳を直した。次の行から効かせる（会議を止めない）。"""
        if kind == "glossary":
            use_external = self._live_stt is not None and self._external_cfg.glossary_path
            path = Path(self._external_cfg.glossary_path if use_external else self.meeting.glossary_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            self._glossary = task_hub_dictionary.merge_pairs(load_glossary(path),
                                                          task_hub_dictionary.live_pairs(self._task_hub_pairs))
            self._glossary_protect = load_protected(path)
            logger.info("画面で辞書が直されたので読み直しました: %s（%d 語）", path.name, len(self._glossary))
        elif kind == "voices" and self._voices is not None:
            self._voices = VoiceLibrary(self._voice_library_path(), self._voice_cfg)

    # ----------------------------------------- この会議のクライアント（議事録・タスクの行き先）

    def _client_name(self) -> str:
        return clients.effective(self.session_dir, self._task_hub_cfg.client_name)[0]

    def client_info(self) -> dict:
        """画面に出す: いま選ばれている相手と、選べる一覧。"""
        chosen = clients.read(self.session_dir)
        return {"chosen": chosen, "default": self._task_hub_cfg.client_name,
                "options": clients.options(self._clients)}

    def choose_client(self, name: str, client_id: str = "", how: str = "画面") -> dict:
        """この会議のクライアントを決める（議事録・タスク・辞書の行き先が変わる）。"""
        entry = clients.choose(self.session_dir, name, client_id=client_id, how=how)
        if self._bridge is not None:
            self._bridge.set_client(entry["name"])     # タスクの行き先も同じ相手にする
        self._client_decided.set()
        self.bus.publish("client", self.client_info())
        print(f"\n  この会議のクライアント: {entry['name']}（議事録とタスクはこの相手のところへ）")
        return entry

    def _confirm_client(self) -> None:
        """会議の終わりに、クライアントを**引き渡す前に**もう一度聞く。

        初回面談がその場で案件になることがある（運用者 指示 2026-09-16）。選ばれていなければ
        画面で聞き、答えが無ければ自社（自社）へ倒す。黙って 自社 にしない — 何にしたかは必ず出す。
        """
        default = self._task_hub_cfg.client_name
        name, chosen = clients.effective(self.session_dir, default)
        if chosen:
            print(f"\n  議事録とタスクの行き先: {name}")
            return
        if self._ui_thread is not None:
            self._client_decided.clear()
            self.bus.publish("client", {**self.client_info(), "ask": True,
                                        "timeout_sec": self._external_cfg.ask_timeout_sec})
            print(f"\n  ── この会議はどのクライアントの会議ですか? ──"
                  f"\n    画面で選んでください（{self._external_cfg.ask_timeout_sec:.0f} 秒で "
                  f"{default} として進みます）")
            self._client_decided.wait(self._external_cfg.ask_timeout_sec)
        name, chosen = clients.effective(self.session_dir, default)
        if not chosen:
            print(f"  クライアントは選ばれませんでした。{name}（自社）として引き渡します"
                  f"（あとで直すなら client.json）")

    def _start_screen_watch(self) -> None:
        """画面共有の取り込みを始める（切ってあれば何もしない）。会議は止めない。"""
        if not self._screen_cfg.enabled:
            return
        try:
            binary = build_ocr(Path(__file__).resolve().parent.parent) if self._screen_cfg.ocr else None
            self._screens = ScreenWatcher(self.session_dir, self._screen_cfg, binary)
            self._screens.start(lambda: time.time() - (self._capture_started_at or self._started_at))
            print(f"  画面共有を {self._screen_cfg.interval_sec:.0f} 秒ごとに控えます"
                  f"（会議のウィンドウだけ・手元にだけ残します）")
        except Exception:  # noqa: BLE001 — 画面が取れなくても会議は続ける
            logger.exception("画面共有の取り込みを始められませんでした")
            self._screens = None

    def _sync_task_hub_dictionary(self) -> None:
        """タスク管理 の辞書を読み直してキャッシュする（次の会議と、この会議の候補づくりに使う）。
        読めなくても何も困らない（前のキャッシュのまま）。"""
        if not self._task_hub_sync or self._bridge is None:
            return
        try:
            found = clients.fetch(self._task_hub_cfg.shared_dir)
            clients.save_cache(Path(__file__).resolve().parent.parent / clients.CACHE_FILE, found)
            self._clients = found
            logger.info("クライアントの一覧を読みました: %d 件", len(found))
        except Exception as error:  # noqa: BLE001 — 控えのままでも会議は回る
            logger.warning("クライアントの一覧を読めませんでした（前の控えのまま）: %s", error)
        try:
            name = self._client_name()
            pairs = task_hub_dictionary.fetch(self._task_hub_cfg.shared_dir, name)
            task_hub_dictionary.save_cache(Path(__file__).resolve().parent.parent / task_hub_dictionary.CACHE_FILE,
                                        pairs, name)
            self._task_hub_pairs = pairs
            logger.info("タスク管理 の辞書を読みました: %d 件（会議中に当てる名前の対 %d 件）",
                        len(pairs), len(task_hub_dictionary.live_pairs(pairs)))
        except Exception as error:  # noqa: BLE001
            logger.warning("タスク管理 の辞書を読めませんでした（前のキャッシュのまま）: %s", error)

    def _suggest_glossary(self) -> None:
        """置き換え辞書の候補を出す（手元だけ・LLM なし）。辞書に入れるのは人が選んだものだけ。
        議事録を作ったあと（`scripts/make_minutes.py`）に、議事録の LLM の指摘を足して作り直す。"""
        try:
            from src.text.glossary_candidates import build_for_session, glossary_for_session

            meeting = {"glossary_path": self.meeting.glossary_path, "external_stt": self.meeting.external_stt}
            glossary = glossary_for_session(self.session_dir, meeting, Path(__file__).resolve().parent.parent)
            names = list(self._voices.people) if self._voices is not None else []
            path, candidates = build_for_session(self.session_dir, glossary_path=glossary,
                                                 self_name=self.meeting.self_name, extra_names=names,
                                                 task_hub_pairs=self._task_hub_pairs)
        except Exception:  # noqa: BLE001 — 候補が出なくても会議の記録は失われない
            logger.exception("辞書の候補づくりに失敗")
            return
        if not candidates:
            return
        print(f"  辞書の候補 {len(candidates)} 件（画面「会議アシスタント」で選べます）")
        if self.meeting.open_library_after and self._ui_cfg.enabled and self._ui_cfg.open_browser:
            # 会議の画面は閉じるので、候補を選ぶ画面を別に開く（ファイルを開いて編集しなくていいように）
            script = Path(__file__).resolve().parent.parent / "scripts" / "library_ui.py"
            try:
                with (self.session_dir / "library_ui.log").open("a", encoding="utf-8") as log:
                    subprocess.Popen([sys.executable, str(script), "--session", self.session_dir.name, "--tab", "candidates"],
                                     stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
            except OSError:
                logger.exception("会議アシスタントの画面を開けませんでした")

    def _verify_unavailable_reason(self, claude_found: bool) -> str:
        """🔍 が使えない理由（Gemini に切り替わる前の、いったんの理由）。"""
        if self._verify_engine == "none":
            return "設定で切ってあります（meeting.verify_engine: none）"
        if self._verify_engine == "claude_cli":
            return "Claude CLI が見つかりません（claude コマンド）"
        if self._verify_engine == "gemini" or (not claude_found and self._external_cfg.enabled):
            return "外（Gemini）への送信の承認がまだです"
        return ("Claude CLI が見つかりません。Gemini で調べるには meeting.external_stt.enabled を true にして、"
                "会議の始めに外への送信を承認してください")

    def _wants_gemini_verify(self) -> bool:
        """🔍 を Gemini＋Google 検索で回したいか。Claude CLI が使えるときは、auto ではそちらを優先する。"""
        if self._verify_engine == "gemini":
            return True
        return (self._verify_engine == "auto" and self._verifier.client is None
                and self._external_cfg.enabled)

    def _start_gemini_verify(self, approval) -> bool:
        """🔍 を Gemini＋Google 検索で回す。承認は `_start_external_engines` で取り済み。"""
        known = set(GeminiLlmConfig.__dataclass_fields__)
        try:
            client = GeminiClient(GeminiLlmConfig(**{k: v for k, v in self._llm_gemini.items() if k in known}))
        except (ValueError, OSError) as error:
            self._verifier.disable(f"Gemini の用意ができません（{error}）")
            return False
        self._verify_gemini = client
        self._verifier.use(GeminiVerifyClient(client))
        record_send(self.session_dir, approval, model=client.config.model, note="verify")
        logger.info("🔍 の裏取りは外（%s＋Google 検索）で回します", client.config.model)
        print(f"  🔍 の裏取りは外（{client.config.model}＋Google 検索）で回します。送るのは押した行と前後の文字だけです。")
        return True

    @property
    def _prior_tasks(self) -> str:
        """事前資料の全文（資料が無ければ「未設定」の断り）。"""
        return self._prep.text

    @property
    def _prep_for_prompt(self) -> str:
        """毎窓のプロンプトへ載せるほう（長い資料は 1 回だけまとめたもの＋固有名詞の 1 節）。"""
        return "\n\n".join(part for part in (self._prep.for_prompt, prompt_line(self._meeting_terms)) if part)

    def _refresh_terms(self) -> None:
        """資料から固有名詞を拾い直してセッションに残す。資料が変わるたびに呼ぶ。"""
        text = self._prep.text
        if not text or text.startswith("（前回タスク"):
            self._meeting_terms = []
        else:
            self._meeting_terms = extract_terms(text, extra=self._prep.digest_terms)
        try:
            save_terms(self.session_dir, self._meeting_terms)
        except OSError:
            logger.exception("固有名詞の一覧を残せませんでした")
        if self._meeting_terms:
            logger.info("事前資料の固有名詞（この会議の守り札）: %s", "、".join(self._meeting_terms))

    def _protected_words(self) -> list[str]:
        """置き換えで壊さない語 = 辞書の守り札＋事前資料の固有名詞＋画面で付けた名前。"""
        names = [part for name in self._aliases.values() for part in re.split(r"[\s　]+", name)
                 if len(part) >= 3 and not part.startswith("不明話者")]
        return list(dict.fromkeys([*self._glossary_protect, *self._meeting_terms, *names]))

    # ------------------------------------------------------------ 事前資料

    def load_prep_materials(self, prep_dir: str | Path) -> None:
        """事前資料とシステムプロンプトを読み込む。

        資料そのものの扱いは `src/meeting/prep.py`。ここはシステムプロンプトだけを見る。
        """
        self._prep.load(prep_dir)
        self._prep.for_prompt, _ = self._prep.compress()
        self._refresh_terms()

        prompt_name = "meeting_state_system.md" if self.meeting.summary_mode == "rolling" else "meeting_system.md"
        repo_path = Path(__file__).resolve().parent.parent
        for candidate in (repo_path / "prompts" / prompt_name, repo_path / "prompts" / "summary_system.md"):
            if candidate.exists():
                self._system_prompt = candidate.read_text(encoding="utf-8").strip()
                logger.info("System prompt: %s", candidate)
                break
        else:
            logger.warning("システムプロンプトが見つかりません")
            self._system_prompt = "あなたは会議支援AIです。会話を要約してください。"

    # -------------------------------------------------- 画面から事前資料を足す
    #
    # 中身は `src/meeting/prep.py`。ここは画面（`UiActions`）から呼ばれる口だけを残す。

    def prep_dir(self) -> Path:
        """この会議の事前資料を置く場所（セッションの中）。会議ごとに空から始まる。"""
        return self._prep.directory()

    def prep_files(self) -> list[dict]:
        return self._prep.files()

    def prep_status(self) -> dict:
        return self._prep.status()

    def attach_prep_file(self, name: str, data: bytes) -> dict:
        """画面から**ファイル**を足す（PDF・Word・Excel・PowerPoint も）。"""
        return self._prep.attach_file(name, data)

    def attach_prep(self, name: str, text: str) -> dict:
        """画面から事前資料を足す。会議の途中でも足せる（次の状態更新から効く）。"""
        return self._prep.attach(name, text)

    def remove_prep(self, name: str) -> dict:
        """足した資料を外す（間違えて付けたとき）。"""
        return self._prep.remove(name)

    def _on_prep_changed(self) -> None:
        """資料が変わったときに、**動いている要約と文字起こしへその場で渡す**。"""
        self._refresh_terms()
        if self._state_loop is not None:
            # 次の窓のプロンプトから効く（`meeting.state.interval_sec`・既定 2 分以内）。
            #   すぐ効かせたいときは画面の「いま見直す」
            self._state_loop.updater.prior_tasks = self._prep_for_prompt
        if self._stt_engine == "local":
            # 手元の Whisper だけがヒントを使う（外のエンジンには効かない）
            self.whisper.config.initial_prompt = self._build_initial_prompt()

    # ------------------------------------------------------------ 起動

    def run(self) -> None:
        """メインループを開始する。Ctrl+C で停止。"""
        if self._llm_engine != "gemini" and not self.llm.health_check():
            logger.error("Ollama に接続できません。ollama serve が起動しているか確認してください。")
            return

        # デバイス解決は最初にやる。ここで失敗したら何もしないで止まる。
        self.mic = dev.resolve_input_device(self.meeting.mic_candidates, label="自分のマイク")
        if self.meeting.capture_backend == "screencapturekit":
            if not has_screen_capture_permission():
                # 案内だけでなく許可も要求する（一覧から項目が消えていても載せ直す・2026-09-24）
                logger.error(permission_help(responsible_app_name()))
                return
        elif self.meeting.capture_backend == "blackhole":
            self.loopback = dev.resolve_input_device(self.meeting.loopback_candidates, label="ループバック")
        else:
            raise ValueError(f"未対応の capture_backend です: {self.meeting.capture_backend}")

        if self._ui_cfg.enabled:
            try:
                self._ui_thread = start_ui_server(
                    build_app(self.bus, self, self.session_dir, library_router=self._library_router()), self._ui_cfg)
                # 塞がっていたら隣のポートで開く（会議を止めない）。実際のポートで案内する
                self._ui_port = int(getattr(self._ui_thread, "port", self._ui_cfg.port))
                logger.info("ローカル UI を起動しました: http://%s:%d/", self._ui_cfg.host, self._ui_port)
                if self._ui_port != self._ui_cfg.port:
                    print(f"\n  ※ ポート {self._ui_cfg.port} が塞がっていたので、画面は "
                          f"http://{self._ui_cfg.host}:{self._ui_port}/ で開きます\n")
                if self._ui_cfg.open_browser:
                    webbrowser.open(f"http://{self._ui_cfg.host}:{self._ui_port}/")
            except PortInUse as exc:
                # 前のセッションが残っている。音声処理は続けるが、UI が「別のセッション」を見せる事故を防ぐため大きく出す
                logger.error("%s", exc)
                print(f"\n  ✗ {exc}\n  （UI 無しで続行します。字幕は Markdown と transcripts.jsonl に出ます）\n")
            except Exception:
                logger.warning("ローカル UI の起動に失敗しましたが、音声処理は続行します", exc_info=True)

        self.writer.interview_purpose = self._extract_purpose()
        self.writer.init_files(self._prior_tasks)

        if self._stt_engine == "local":
            logger.info("Whisper モデルを事前ロード中…")
            self.whisper._ensure_model()
            logger.info("Whisper モデルロード完了。")
        else:
            # 外で起こす予定なのでモデルは読まない（GPU を Ollama に明け渡す。方式④の要点）。
            #   関門を通らなかったときは `_fall_back_to_local()` がその場で読む。
            logger.info("会議中の文字起こしは外（%s）で回す予定です。Whisper は読みません。",
                        self._external_cfg.model)

        self._start_capture()
        self._start_screen_watch()
        # 外へ出す準備（課金の確認・キー）は、測る前に見ておく。切れていても会議は始められるが、
        #   「外へ出す」を選んだのに手元に落ちる、という分かりにくい状態を先に知らせる
        self.startup_warnings = list(self.startup_warnings) + self.external_readiness()
        if not self._preflight():
            logger.error("セルフチェックで中断しました。")
            self._abort_startup()
            return

        # 話者登録（相手側のみ）
        speakers_path = self.session_dir / "speakers.json"
        self.bus.publish("status", {"phase": "enrolling", "warnings": []})
        if self.meeting.skip_enrollment:
            from src.enrollment import load_or_create

            self.diarizer, _ = load_or_create(speakers_path, self._enrollment_config())
        else:
            self.diarizer = run_enrollment(
                CaptureFrameSource(self.capture, REMOTE_KEY, self._audio_cfg.sample_rate),
                speakers_path=speakers_path,
                config=self._enrollment_config(),
            )
        if self.capture is not None:
            # 本編前の直近ぶん（挨拶・雑談）は残す。そこで話者に名前を付ける運用のため
            for key in (SELF_KEY, REMOTE_KEY):
                dropped = self.capture.trim_backlog(key, self.meeting.keep_backlog_sec)
                kept = self.meeting.keep_backlog_sec
                logger.info("本編前の音声: %s は直近 %.0f 秒を残し、%.0f 秒を捨てました", key, kept, dropped)
            self._align_clocks_to_recording()

        self.whisper.config.initial_prompt = self._build_initial_prompt()

        self._start_external_engines()

        if self.meeting.summary_mode == "rolling":
            updater = StateUpdater(
                self.llm,
                self._system_prompt,
                self._prep_for_prompt or self._prior_tasks,
                self._state_cfg,
            )
            self._state_loop = StateLoop(updater, on_state=self._on_state)
            self._state_loop.start()

        if self._factcheck_cfg.enabled:
            self._factcheck_loop = FactCheckLoop(FactChecker(self.llm, self._factcheck_cfg), on_claim=self._on_claim)
            self._factcheck_loop.start()
            logger.info("ファクトチェックを有効にしました（判定: %s）", self._factcheck_cfg.verify)

        self.bus.publish("status", {"phase": "running", "warnings": []})
        logger.info("会議の記録を開始しました。")

        last_summary_time = time.time()
        try:
            while True:
                if self._finish_requested.is_set():
                    logger.info("画面から終了を受け取りました — シャットダウン中…")
                    break
                self._process_audio_queues()
                self._maybe_auto_review(time.time())

                if self.meeting.summary_mode == "legacy" and time.time() - last_summary_time >= self._summary_interval:
                    with self._buffer_lock:
                        buf_size = len(self._transcript_buffer)
                    if buf_size > 0:
                        logger.info("=== 要約サイクル開始 (segments=%d) ===", buf_size)
                        self._run_summary_cycle()
                    else:
                        logger.info("バッファ空のため要約スキップ")
                    last_summary_time = time.time()

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Ctrl+C 受信 — シャットダウン中…")
        finally:
            self._shutdown(speakers_path)

    def _enrollment_config(self) -> EnrollmentConfig:
        return EnrollmentConfig(
            duration_sec=self.meeting.enroll_duration_sec,
            sample_rate=self._audio_cfg.sample_rate,
            similarity_threshold=self.meeting.similarity_threshold,
            max_adapt_embeddings=self.meeting.max_adapt_embeddings,
            diarizer=self._diarizer_cfg,
        )

    # ------------------------------------------------------ セルフチェック

    def external_readiness(self) -> list[str]:
        """外へ出す準備ができているかを、**会議が始まる前に**確かめる。

        2026-09-14 に踏んだ: `gcloud` の認証は黙って切れる。切れていると関門が
        「確認できない＝送らない」に倒れるので、画面で「外へ出す」を選んでも**手元へ落ちる**
        （会議は失われないが、画面が 17 分遅れる。その場では気づきにくい）。
        ∴ 起動セルフチェックで先に言う。直し方（`gcloud auth login`）まで出す。

        戻り値は警告の行。準備ができていれば空。
        """
        wants = [name for name, engine in (("文字起こし", self._stt_engine), ("要約", self._llm_engine))
                 if engine in EXTERNAL_STT_ENGINES]
        if not wants:
            return []                       # そもそも外へ出さない設定なので点検しない
        label = "・".join(wants)
        config = self._external_cfg
        # 「何が起きるか」まで言う。理由だけだと、画面を見た人が困り方を想像できない
        effects = []
        if "文字起こし" in wants:
            effects.append("文字起こしが画面に出るまで 17 分ほど遅れます")
        if "要約" in wants:
            effects.append("要約は手元の Ollama で回します")
        fallback = f"このまま始めると{label}は手元で処理します（{'／'.join(effects)}）"
        if not config.enabled:
            return [f"{label}を外で回す設定ですが、外部送信が切ってあります（external_stt.enabled）。{fallback}"]
        if not config.billing_project:
            return [f"{label}を外で回す設定ですが、送り先のプロジェクト ID がありません。{fallback}"]
        if config.key_file and not Path(config.key_file).expanduser().exists():
            return [f"{label}を外で回す設定ですが、API キーのファイルがありません"
                    f"（{config.key_file}）。{fallback}"]
        ok, reason = billing_enabled(config.billing_project, timeout_sec=EXTERNAL_CHECK_TIMEOUT_SEC)
        if not ok and reason == LOGIN_EXPIRED and config.auto_login:
            # 端末で打たなくていいように、ログイン画面をこちらから開いて待つ（会議が始まる前に済ませる）
            def say(message: str) -> None:
                print(f"\n  {message}")
                self.bus.publish("status", {"phase": "starting", "warnings": [message]})
            ok, reason = login_and_wait(config.billing_project, timeout_sec=config.login_timeout_sec, on_status=say)
        if not ok:
            return [f"{label}を外へ出す準備ができていません: {reason}。{fallback}"]
        logger.info("外へ出す準備は問題ありません（%s / %s）", label, config.billing_project)
        return []

    def _preflight(self) -> bool:
        """本編前のセルフチェック。

        「録れているつもりで無音」が本システム最大の事故なので、必ずここを通す。
        戻り値 False なら中断する。

        2026-09-12: 画面（ダッシュボード）で見て、画面で決められるようにした（運用者 依頼
        「ターミナル操作をなるべく避けたい」）。UI が動いていないときだけ、従来どおり端末で聞く。
        """
        while True:
            result = self._measure_preflight()
            if result.get("cancelled"):
                return False
            if not result["warnings"]:
                print("  ✅ 問題は見つかりませんでした。")
                self.bus.publish("preflight", {**result, "phase": "ok"})
                return True
            print("-" * 66)
            for warning in result["warnings"]:
                print(f"  ⚠ {warning}")
            print("-" * 66)
            decision = self._ask_preflight(result)
            if decision == "retry":
                print("\n  もう一度測ります…")
                continue
            return decision == "start"

    def _abort_startup(self) -> None:
        """本編に入らずに終わるときの後片付け。

        2026-09-12: 中止を選ぶと macOS が「Python が予期しない理由で終了しました」を出していた。
        音声の取り込み（ScreenCaptureKit / sounddevice）と UI を止めずに main を抜けていたため。
        """
        print("\n  中止しました。後片付けをしています…")
        if self._ui_thread is not None:
            self._ui_thread.server.should_exit = True
        if self.capture is not None:
            try:
                self.capture.stop()
            except Exception:  # noqa: BLE001 — 片付けで落ちない
                logger.debug("キャプチャの停止に失敗", exc_info=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
        try:
            self.llm.close()
        except Exception:  # noqa: BLE001
            logger.debug("Ollama クライアントの終了に失敗", exc_info=True)
        print("  終了しました。もう一度やり直すときは、また起動してください。")

    def _ask_preflight(self, result: dict) -> str:
        """警告があったときに「開始／測り直し／中止」を聞く。画面が使えるなら画面で。"""
        if self._ui_thread is None:
            answer = input("この状態で開始しますか [y/N]: ").strip().lower()
            return "start" if answer in {"y", "yes", "は", "はい"} else "cancel"
        self._preflight_answer = None
        self._preflight_decided.clear()
        self.bus.publish("preflight", {**result, "phase": "ask"})
        print("\n  画面（ダッシュボード）で「このまま開始」「もう一度測る」「中止」を選んでください。")
        while not self._preflight_decided.wait(timeout=30):
            print("  … 画面の選択を待っています。")
        return self._preflight_answer or "cancel"

    def _ask_external_send(self, files: list[Path], config) -> bool:
        """録音を外へ出してよいか聞く。画面が使えるなら画面で。

        返事が無ければ送らない（会議後の作り直しは無人で走ることがある）。
        画面も端末も無ければ送らない。聞けない＝承認されていない。
        """
        payload = {
            "phase": "ask",
            "session": self.session_dir.name,
            "files": [path.name for path in files],
            "minutes": round(sum(audio_minutes(path) for path in files)),
            "project": config.billing_project,
            "timeout_sec": config.ask_timeout_sec,
        }
        return self._ask_external(
            payload,
            f"\n  録音を外（Google・{config.billing_project}）へ出しますか?",
            config.ask_timeout_sec,
        )

    def _ask_external_live(self, config, items: list[str] | None = None) -> bool:
        """会議**中**の音声を 30 秒ごとに外へ出してよいか、会議が始まる前に聞く。

        会議のあとの作り直し（`_ask_external_send`）とは別に、毎回ここでも聞く。
        「一度 on にしたら以後ずっと」にしない、が 09-13 の決めごと（`CLAUDE.md`）。
        聞けない（画面も端末も無い）・返事が無い＝**承認されていない**ので手元で起こす。
        """
        payload = {
            "phase": "ask",
            "session": self.session_dir.name,
            "files": list(items or ["会議中の音声（30 秒ごと）"]),
            "minutes": 0,
            "project": config.billing_project,
            "timeout_sec": config.ask_timeout_sec,
            "live": True,
        }
        return self._ask_external(
            payload,
            f"\n  この会議の「{'」「'.join(items or ['会議中の音声（30 秒ごと）'])}」を"
            f"、外（Google・{config.billing_project}）へ出しますか?",
            config.ask_timeout_sec,
        )

    def _gap_entries(self) -> list[dict]:
        """会議中に外へ送れなかった区間（拾い直しの対象）。"""
        path = self.session_dir / GAPS_FILE
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _ask_external_gaps(self, config, gaps: list[dict]) -> bool:
        """送れなかった区間だけを、もう一度外へ出してよいか聞く。

        会議中の承認とは別に聞く（送るものが変わるため）。断られたら手元の Whisper で拾う
        ＝ どちらにしても議事録は埋まる。
        """
        seconds = sum(float(gap.get("end_time", 0.0)) - float(gap.get("start_time", 0.0)) for gap in gaps)
        payload = {
            "phase": "ask",
            "session": self.session_dir.name,
            "files": [f"送れなかった {len(gaps)} 区間（合計 {seconds:.0f} 秒）"],
            "minutes": round(seconds / 60),
            "project": config.billing_project,
            "timeout_sec": config.ask_timeout_sec,
            "live": True,
        }
        return self._ask_external(
            payload,
            f"\n  会議中に送れなかった {len(gaps)} 区間（{seconds:.0f} 秒）を外へ出して拾い直しますか?"
            "\n  （出さない場合は手元の Whisper で拾います）",
            config.ask_timeout_sec,
        )

    def _ask_external(self, payload: dict, question: str, timeout_sec: float) -> bool:
        """画面（無ければ端末）で「外へ出す／手元で作る」を聞く。返事が無ければ False。"""
        if self._ui_thread is None:
            print(question)
            try:
                answer = input("  出してよければ y / 手元で作るなら N [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            return answer in {"y", "yes", "は", "はい"}

        self._external_answer = None
        self._external_decided.clear()
        self.bus.publish("external_send", payload)
        print(f"\n  画面（ダッシュボード）で「外へ出す」「手元で作る」を選んでください"
              f"（{timeout_sec:.0f} 秒で手元に倒れます）。")
        if not self._external_decided.wait(timeout=timeout_sec):
            self.bus.publish("external_send", {**payload, "phase": "done", "decision": "keep"})
            print("  返事が無いので手元で作ります。")
            return False
        decided = self._external_answer == "send"
        self.bus.publish("external_send", {**payload, "phase": "done",
                                           "decision": self._external_answer or "keep"})
        return decided

    def answer_external_send(self, decision: str) -> dict:
        """画面からの「外へ出す／手元で作る」を受け取る。"""
        if decision not in {"send", "keep"}:
            raise ValueError(f"未知の選択: {decision}")
        self._external_answer = decision
        self._external_decided.set()
        return {"ok": True, "decision": decision}

    def answer_preflight(self, decision: str) -> dict:
        """画面からのセルフチェックの回答を受け取る。"""
        if decision not in {"start", "retry", "cancel", "measure", "skip", "start_now"}:
            raise ValueError(f"未知の選択: {decision}")
        self._preflight_answer = decision
        self._preflight_decided.set()
        return {"ok": True, "decision": decision}

    def _measure_preflight(self) -> dict:
        """機器とレベルを実測して、結果と警告を返す（表示と判断は呼び出し側）。

        測定はボタンを押してから始める（運用者 依頼 2026-09-12「自分の発話 3 秒がシビア」）。
        自動で始まると、身構える前に 3 秒が終わって「無音」と判定される。
        """
        print()
        print("=" * 66)
        print("【起動セルフチェック】")
        print("=" * 66)
        print(f"  自分のマイク : {self.mic}  ← 候補 {self.mic.matched_candidate!r}")
        print(f"  相手側取得   : {self.loopback if self.loopback else 'ScreenCaptureKit'}")
        print()
        base = {
            "mic": str(self.mic),
            "mic_name": self.mic.name,
            "remote": str(self.loopback) if self.loopback else "ScreenCaptureKit",
            "seconds": self.meeting.level_check_sec,
        }
        warnings = list(self.startup_warnings)
        if self.meeting.capture_backend == "blackhole":
            warnings += self._check_output_routing()

        # --- 自分のマイク（ボタンを押してから測る）
        action = self._wait_preflight_action("wait_mic", {**base, "warnings": warnings})
        if action == "cancel":
            return {**base, "cancelled": True, "warnings": warnings}
        if action == "start_now":
            # 会議がもう始まっている場合の急ぎ口（2026-09-12 運用者）。測定は飛ばし、
            #   代わりに開始後の無音見張り（_watch_silence）が「録れているつもりで無音」を捕まえる。
            print("  測定を飛ばして本編に入ります（開始後に無音を見張ります）。")
            return {**base, "skipped": True, "warnings": []}
        mic_level = None
        if action != "skip":
            print(f"  マイクを {self.meeting.level_check_sec:g} 秒測ります — 話してください…")
            self.bus.publish("preflight", {**base, "phase": "mic", "warnings": warnings})
            mic_level = dev.level_of(self.capture.record_from(SELF_KEY, self.meeting.level_check_sec),
                                     self.meeting.level_check_sec)
            print(f"    自分のマイク : {mic_level}")
            warnings += self._level_warnings("自分のマイク", self.mic.name, mic_level)
        else:
            print("  マイクの測定を飛ばしました。")

        # --- 相手側（同じくボタンを押してから）
        measured = {**base, "mic_dbfs": round(mic_level.peak_dbfs, 1) if mic_level else None}
        action = self._wait_preflight_action("wait_remote", {**measured, "warnings": warnings})
        if action == "cancel":
            return {**measured, "cancelled": True, "warnings": warnings}
        if action == "start_now":
            print("  測定を飛ばして本編に入ります（開始後に無音を見張ります）。")
            return {**measured, "skipped": True, "warnings": []}
        remote_level = None
        if action != "skip":
            print(f"  相手側を {self.meeting.level_check_sec:g} 秒測ります — 相手が話している状態にしてください…")
            self.bus.publish("preflight", {**measured, "phase": "remote", "warnings": warnings})
            remote_level = dev.level_of(self.capture.record_from(REMOTE_KEY, self.meeting.level_check_sec),
                                        self.meeting.level_check_sec)
            print(f"    相手側取得 : {remote_level}")
            if remote_level.is_empty:
                warnings.append(
                    f"相手側（{measured['remote']}）から音声が1フレームも届いていません。"
                    "画面収録の許可を確認してください（許可が無いと無音のまま進みます）。"
                )
            elif remote_level.is_silent and self.meeting.capture_backend == "blackhole":
                warnings.append(
                    f"ループバックが無音です。{self.meeting.app_name} の「スピーカー」が "
                    f"『{self.meeting.multi_output_name}』になっているか確認してください。"
                    "（相手が誰も話していないだけなら問題ありません）"
                )
        else:
            print("  相手側の測定を飛ばしました。")

        if (self.meeting.capture_backend == "screencapturekit" and self.meeting.beep_selftest
                and (remote_level is None or remote_level.is_silent)):
            # 相手側に音が届いていれば SCK は生きている。届いていないときだけビープで確かめる
            if not self._beep_selftest():
                app = responsible_app_name()
                warnings.append(f"ScreenCaptureKit にシステム音声が届いていません。画面収録の許可先『{app}』を確認してください。")

        print()
        return {
            **measured,
            "remote_dbfs": round(remote_level.peak_dbfs, 1) if remote_level else None,
            "warnings": warnings,
        }

    def _level_warnings(self, label: str, device_name: str, level) -> list[str]:
        """測ったレベルから警告文を作る。「1フレームも来ない」と「無音」は原因が違う。"""
        if level.is_empty:
            return [f"{label}（{device_name}）から音声が1フレームも届いていません。"
                    "機器が外れていないか、ほかのアプリが専有していないか確認してください。"]
        if level.is_silent:
            return [f"{label}（{device_name}）から音が来ていません。"
                    "話していたなら、ミュートになっていないか、機器が正しく繋がっているか確認してください。"]
        return []

    def _wait_preflight_action(self, phase: str, payload: dict) -> str:
        """画面のボタン（測る／飛ばす／中止）を待つ。画面が無ければ Enter を待つ。"""
        prompts = {"wait_mic": "自分のマイクを測ります", "wait_remote": "相手側を測ります"}
        if self._ui_thread is None:
            answer = input(f"  {prompts.get(phase, '測ります')} — 準備ができたら Enter（s=飛ばす / q=中止）: ").strip().lower()
            return {"s": "skip", "q": "cancel"}.get(answer, "measure")
        self._preflight_answer = None
        self._preflight_decided.clear()
        self.bus.publish("preflight", {**payload, "phase": phase})
        print(f"  {prompts.get(phase, '測定')}: 画面のボタンを押してください。")
        while not self._preflight_decided.wait(timeout=30):
            print("  … 画面の操作を待っています。")
        return self._preflight_answer or "cancel"

    def _beep_selftest(self) -> bool:
        """ビープを再生し、ScreenCaptureKit がその音を受けたか確認する。"""
        samples = int(round(self.meeting.beep_sec * self._audio_cfg.sample_rate))
        time_axis = np.arange(samples, dtype=np.float32) / self._audio_cfg.sample_rate
        amplitude = float(10 ** (self.meeting.beep_dbfs / 20.0))
        tone = amplitude * np.sin(2 * np.pi * self.meeting.beep_hz * time_axis)
        import wave
        beep_path = self.session_dir / "beep.wav"
        with wave.open(str(beep_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._audio_cfg.sample_rate)
            wav.writeframes((tone * 32767).astype(np.int16).tobytes())
        # afplay は再生完了まで戻らないので、先に再生を始めてから「これから 1.5 秒」を録る
        #   （順に呼ぶと録り始めた時点でビープが終わっている）。record_from は溜まりを捨ててから取る
        player = subprocess.Popen(["afplay", str(beep_path)])
        try:
            audio = self.capture.record_from(REMOTE_KEY, 1.5)
        finally:
            player.wait(timeout=10)
        result = detect_tone(audio, self._audio_cfg.sample_rate, self.meeting.beep_hz)
        ok = result.tone_sec >= self.meeting.beep_sec * 0.5
        self.bus.publish(
            "status",
            {
                "phase": "beep_selftest",
                "ok": ok,
                "tone_sec": result.tone_sec,
                "peak_dbfs": result.peak_dbfs,
                "dominant_hz": result.dominant_hz,
            },
        )
        logger.info(
            "ビープ自己診断: ok=%s tone=%.2fs peak=%.1f dBFS dominant=%.1f Hz",
            ok, result.tone_sec, result.peak_dbfs, result.dominant_hz,
        )
        return ok

    def _check_output_routing(self) -> list[str]:
        """会議アプリの出力系統を点検して、警告文のリストを返す。"""
        warnings: list[str] = []

        current_output = dev.default_output_name()
        target = self.meeting.multi_output_name
        if current_output:
            print(f"  既定の出力   : {current_output}")
            if _normalize(target) not in _normalize(current_output):
                warnings.append(
                    f"macOS の既定出力が『{current_output}』です。"
                    f"{self.meeting.app_name} の「スピーカー」を『{target}』に指定しないと、"
                    "ループバックに何も入りません。"
                )

        info = dev.read_multi_output_info(target)
        if info is None:
            print(f"  複数出力装置 : （構成を読めませんでした: {target}）")
            return warnings

        paired = dev.paired_bluetooth_macs()

        print(f"  複数出力装置 : {info.name}")
        for i, (sub, state) in enumerate(info.classified(paired), 1):
            mark = {dev.GHOST: "✗", dev.OFFLINE: "･", dev.ONLINE: " "}[state]
            drift = "drift補正ON " if sub.drift_correction else "drift補正OFF"
            label = sub.name or "(オフライン装置)"
            role = "（プライマリ）" if i == 1 else ""
            print(f"     {mark} {drift}  {label}{role}")

        if not info.contains(self.loopback.name):
            warnings.append(
                f"『{info.name}』に {self.loopback.name} が入っていません。"
                "この構成では録音できません。"
            )

        # 「自分の耳に届く経路」があるか。
        #   マイクと聴くデバイスは別でよい（例: Shure MV7 で喋り、有線イヤホンで聴く）ので、
        #   「今日のマイクが束に入っているか」で代用してはいけない。
        audible = info.audible_outputs([self.loopback.name])
        if not audible:
            warnings.append(
                f"『{info.name}』に、今つながっている再生デバイスが1台もありません。"
                f"このままだと {self.meeting.app_name} の音がどこにも聞こえません。"
                "Audio MIDI 設定で、今日聴くデバイスを追加してください。"
            )
        else:
            print(f"  聴こえる経路 : {'、'.join(s.name for s in audible)}")

        # 今日のマイクに対応する「聴く側」が束にあるか
        listening = self.meeting.listening_device_for.get(self.mic.name, self.mic.name)
        if listening != self.mic.name:
            print(f"  今日の聴く側 : {listening}（マイクとは別系統）")
        if not info.contains(listening):
            warnings.append(
                f"今日のマイク『{self.mic.name}』に対応する聴く側『{listening}』が"
                f"『{info.name}』に入っていません。{self.meeting.app_name} の音が自分に聞こえません。"
            )

        ghosts = info.ghosts(paired)
        if ghosts:
            ghost_uids = ", ".join(g.uid for g in ghosts)
            warnings.append(
                f"『{info.name}』に二度と戻らないサブデバイスが残っています（{ghost_uids}）。"
                "ペアリング済み Bluetooth 一覧にも無い機器です。Audio MIDI 設定から外してください。"
            )

        # stacked 出力ではプライマリ1台以外は drift 補正を入れるのが定石。
        # 補正 OFF が2台以上あると、どれかが必ずドリフトする。
        # オフラインのサブデバイスは drift を設定できないので数えない。
        no_drift = [s for s in info.online() if not s.drift_correction]
        if len(no_drift) >= 2:
            names = "、".join(s.name for s in no_drift)
            warnings.append(
                f"『{info.name}』で drift（音ずれ）補正が OFF のデバイスが2台以上あります（{names}）。"
                "プライマリ1台を除いて補正を ON にしないと、音がずれていきます。"
            )

        return warnings

    # ------------------------------------------------------------ 音声処理

    def _start_capture(self) -> None:
        """2本のストリームを開き、時刻を揃えて ChunkBuilder を組む。"""
        remote = SourceConfig(
            key=REMOTE_KEY,
            device=None,
            device_name="ScreenCaptureKit",
            channels=1,
            speaker=None,
            backend="screencapturekit",
        ) if self.meeting.capture_backend == "screencapturekit" else SourceConfig(
            key=REMOTE_KEY,
            device=self.loopback.index,
            device_name=self.loopback.name,
            channels=min(self.loopback.channels, 2),
            speaker=None,
        )
        sources = [
            SourceConfig(
                key=SELF_KEY,
                device=self.mic.index,
                device_name=self.mic.name,
                channels=1,
                speaker=self.meeting.self_name,
            ),
            remote,
        ]
        self.capture = MultiCapture(sources, self._audio_cfg, record_dir=self.session_dir, sck=self.meeting.sck)
        self.capture.start(first_frame_timeout=self.meeting.first_frame_timeout_sec)

        if not self.capture.wait_for_first_frames(timeout=self.meeting.first_frame_timeout_sec):
            missing = self.capture.silent_sources()
            logger.warning("最初のフレームが届いていないソースがあります: %s", missing)

        offsets = self.capture.start_offsets()
        logger.info(
            "ストリーム開始オフセット: %s",
            ", ".join(f"{k}=+{v * 1000:.0f}ms" for k, v in sorted(offsets.items())),
        )

        # 文字起こしの時刻（start_time）は録音の先頭からの経過。画面と記録で「何時の発言か」を
        #   出すには、その先頭が何時だったかが要る（2026-09-14: 到着時刻を出していて、30 秒の窓
        #   ぶんが全部同じ時刻に見えた）。
        self._capture_started_at = time.time()
        self._builders = {
            SELF_KEY: ChunkBuilder(
                self.meeting.self_name,
                self._vad_cfg,
                self._audio_cfg.sample_rate,
                session_start_offset=offsets.get(SELF_KEY, 0.0),
            ),
            REMOTE_KEY: ChunkBuilder(
                REMOTE_KEY,  # 実際の話者名は identify() の結果で差し替える
                self._vad_cfg,
                self._audio_cfg.sample_rate,
                session_start_offset=offsets.get(REMOTE_KEY, 0.0),
            ),
        }

    def review(self, reason: str = "手動") -> dict:
        """ここまでの全文を Claude CLI に読み直させ、決定事項・TODO を拾い直す。

        会議中に走らせられる（Claude CLI はネットワーク越しで、文字起こしの GPU を奪わない）。
        運用者 の案 2026-09-12: 「会議終了 10 分前に一度まわす」。終わってからでは、その場で
        「決まったこと」を確認できない。30〜60 秒の窓では数分にまたがる合意が見えないので、
        会議中に一度だけでも全文を通すと拾える（2026-09-11 の実測: 決定 0 件 → 16 件）。
        """
        if self._state_loop is None:
            return {"ok": False, "reason": "要約がまだ動いていません"}
        if not self._review_lock.acquire(blocking=False):
            return {"ok": False, "reason": "前回の見直しがまだ動いています"}
        try:
            client = ClaudeCliClient()
            if not client.available():
                return {"ok": False, "reason": "Claude CLI が見つかりません"}
            started = time.time()
            self.bus.publish("status", {"phase": "reviewing", "warnings": [], "note": f"全文を見直しています（{reason}）"})
            before = self._state_loop.state()
            transcripts = self._read_transcripts()
            delta = client.final_pass(before, self._format_transcript(transcripts), final_pass_prompt())
            state, changes = apply_delta(before, delta, before.updated_at)
            with self._state_loop._lock:   # 会議中の更新と競合させない
                self._state_loop._state = state
            self.writer.write_state(state)
            self.writer.write_state_json(state)
            self.bus.publish("state", {"state": state.to_dict(), "latency_sec": time.time() - started, "parse_ok": True})
            self.bus.publish("status", {"phase": "running", "warnings": [],
                                        "note": f"見直し完了（{reason}）: 決定 {len(state.decisions)} 件 / TODO {len(state.todos)} 件"})
            logger.info("見直し（%s）: %.0f 秒 / 変更 %d 件 / 決定 %d 件 / TODO %d 件",
                        reason, time.time() - started, len(changes), len(state.decisions), len(state.todos))
            return {"ok": True, "changes": len(changes), "decisions": len(state.decisions),
                    "todos": len(state.todos), "seconds": round(time.time() - started, 1)}
        except Exception as exc:  # noqa: BLE001 — 会議を止めない
            logger.warning("見直しに失敗しました: %s", exc)
            self.bus.publish("status", {"phase": "running", "warnings": [f"見直しに失敗しました: {exc}"]})
            return {"ok": False, "reason": str(exc)}
        finally:
            self._review_lock.release()

    def finish(self) -> dict:
        """画面から会議を終わらせる（ターミナルで Ctrl+C を押さずに済むように）。"""
        self._finish_requested.set()
        logger.info("終了要求を受け取りました（画面）")
        return {"ok": True, "note": "終了処理に入ります（録音からの作り直しが続きます）"}

    def set_planned_minutes(self, minutes: float | None) -> dict:
        """会議の予定時間を受け取り、終了の少し前に自動で見直す時刻を決める。"""
        self._planned_minutes = minutes
        self._auto_review_at = None
        if minutes:
            self._auto_review_at = self._started_at + max(0.0, minutes - self.meeting.review_lead_min) * 60
        return self.review_plan()

    def review_plan(self) -> dict:
        """予定時間と、自動の見直しが走る時刻を返す（UI 表示用）。"""
        return {
            "planned_minutes": self._planned_minutes,
            "lead_min": self.meeting.review_lead_min,
            "at": self._auto_review_at,
            "done": self._auto_review_done,
        }

    def _maybe_auto_review(self, now: float) -> None:
        """予定時間から決めた時刻になったら、一度だけ自動で見直す。"""
        if self._auto_review_done or self._auto_review_at is None or now < self._auto_review_at:
            return
        self._auto_review_done = True
        threading.Thread(target=self.review, args=(f"終了 {self.meeting.review_lead_min:g} 分前",), daemon=True).start()

    def _on_claim(self, claim: Claim) -> None:
        """拾った主張を画面へ流し、セッションに残す。"""
        self.bus.publish("factcheck", claim.to_dict())
        with (self.session_dir / "factcheck.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
        logger.info("ファクトチェック %s [%s] %s — %s", claim.id, claim.verdict, claim.quote, claim.note)

    def _watch_silence(self, now: float, level_frames: dict[str, list[np.ndarray]]) -> None:
        """本編に入ってから一定時間まったく音が来ないソースを警告する。

        起動セルフチェックは「開始時の一瞬」しか見ない。会議の途中でマイクが外れても、
        あわてて起動して測定を飛ばしたときも、こちらが「録れているつもりで無音」を捕まえる。
        """
        labels = {SELF_KEY: f"自分のマイク（{self.mic.name if self.mic else '不明'}）",
                  REMOTE_KEY: f"相手側（{self.meeting.app_name}）"}
        for key, frames in level_frames.items():
            if frames and self._frames_dbfs(frames) > -70.0:
                self._last_sound_at[key] = now
                if key in self._silence_alerted:
                    self._silence_alerted.discard(key)
                    self.bus.publish("status", {"phase": "running", "warnings": [],
                                                "note": f"{labels[key]} の音が戻りました"})
                continue
            since = self._last_sound_at.get(key) or self._started_at
            if now - since >= self.meeting.silence_alert_sec and key not in self._silence_alerted:
                self._silence_alerted.add(key)
                message = (f"{labels[key]} から {int((now - since) / 60)} 分以上音が来ていません。"
                           "機器・ミュート・画面収録の許可を確認してください。")
                logger.warning("%s", message)
                self.bus.publish("status", {"phase": "running", "warnings": [message]})

    def _log_throughput(self, now: float) -> None:
        """1分ごとに、処理が実時間に追いつけているかをログに残す。

        2026-09-11 本番は 0.7 倍速までしか出ず 17 分遅れたが、同じ音声を会議の外で流すと 10 倍速だった。
        Zoom も要約との GPU 取り合いも原因ではないと分かったので、本番中の数字を残して切り分ける。
        """
        work = list(self._stt_work)
        if not work:
            return
        with self._stt_pending_lock:
            queue_depth = self._stt_pending
        spent = sum(sec for sec, _ in work)
        audio = sum(dur for _, dur in work)
        logger.info(
            "処理状況: 待ち %d 件 / 直近 %d 件は 1 件 %.2f 秒・音声の %.1f 倍速 / 要約 %s / 経過 %.0f 分",
            queue_depth,
            len(work),
            spent / len(work),
            audio / spent if spent else 0.0,
            f"{self._last_state_latency_sec:.0f} 秒" if self._last_state_latency_sec else "未実行",
            (now - self._started_at) / 60,
        )

    def _align_clocks_to_recording(self) -> None:
        """セルフチェックと登録で捨てた音声の分だけ、文字起こしの時計を進める。

        進めないと、文字起こしの時刻が録音ファイルより「捨てた長さ」だけ手前にずれる
        （2026-09-11 本番: 2 回目は登録の 71 秒、1 回目は開始確認で止まっていた 502 秒ずれた。
        MacWhisper の録音と突き合わせて判明）。
        """
        for key, builder in self._builders.items():
            skipped = self.capture.taken_seconds(key)
            builder.skip(skipped)
            logger.info("文字起こしの時計を録音に揃えました: %s +%.1f 秒", key, skipped)

    def _process_audio_queues(self) -> int:
        """両ストリームのキューを掃き、VAD → STT に流す。"""
        count = 0
        level_frames: dict[str, list[np.ndarray]] = {SELF_KEY: [], REMOTE_KEY: []}
        for key, builder in self._builders.items():
            for frame in self.capture.drain(key):
                level_frames[key].append(frame)
                for chunk in builder.feed(frame):
                    self._submit(key, chunk)
                    count += 1
        now = time.time()
        if now - self._last_level_at >= 1.0:
            self.bus.publish("level", {
                "self_dbfs": self._frames_dbfs(level_frames[SELF_KEY]),
                "remote_dbfs": self._frames_dbfs(level_frames[REMOTE_KEY]),
            })
            self._last_level_at = now
        if self._live_stt is not None:
            # 音が途切れたまま溜まっている窓を、壁時計で締める（会議の終わりぎわの取りこぼし対策）
            self._live_stt.tick(now)
        if now - self._last_metrics_at >= 5.0:
            with self._stt_pending_lock:
                queue_depth = self._stt_pending
            if self._live_stt is not None:
                queue_depth = self._live_stt.pending
            self.bus.publish("metrics", {
                "stt_queue_depth": queue_depth,
                "stt_latency_p95_sec": self._p95(self._stt_latencies),
                "state_latency_sec": self._last_state_latency_sec,
                "drift_sec": getattr(self.capture, "drift", lambda: {})().get(REMOTE_KEY),
            })
            self._last_metrics_at = now
        self._watch_silence(now, level_frames)
        if now - self._last_metrics_log_at >= 60.0:
            self._log_throughput(now)
            self._last_metrics_log_at = now
        if now - self._last_drift_log_at >= 300.0:
            logger.info("ドリフト: %s", getattr(self.capture, "drift", lambda: {})())
            self._last_drift_log_at = now
        return count

    @staticmethod
    def _frames_dbfs(frames: list[np.ndarray]) -> float:
        """フレーム列の RMS を dBFS にして、入力なしは -100 を返す。"""
        if not frames:
            return -100.0
        values = np.concatenate([np.asarray(frame, dtype=np.float32).reshape(-1) for frame in frames])
        rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
        return float(max(-100.0, 20.0 * np.log10(max(rms, 1e-5))))

    @staticmethod
    def _p95(values: deque[float]) -> float | None:
        """直近値の p95 を返し、値がなければ None を返す。"""
        return float(np.percentile(list(values), 95)) if values else None

    # ------------------------------------------ 会議中の文字起こしを外で回す（方式④）

    def _start_external_engines(self) -> None:
        """会議中に外（Gemini）で回すものを、**1 回の承認**でまとめて決める。

        送るものは 2 つあり得る。①音声（30 秒ごとの文字起こし） ②全文（会議中の要約）。
        どちらも「その会議について毎回聞く」＝ 設定を on にしただけでは送らない（`CLAUDE.md`）。
        聞けない・断られた・課金を確認できないときは、どちらも手元へ落ちる。
        """
        wants_stt = self._stt_engine in EXTERNAL_STT_ENGINES
        wants_llm = self._llm_engine == "gemini"
        wants_verify = self._wants_gemini_verify()
        if not (wants_stt or wants_llm or wants_verify):
            return
        config = self._external_cfg
        items: list[str] = []
        if wants_stt:
            items.append("会議中の音声（30 秒ごと）")
        if wants_llm:
            # 会議のあとの議事録も、この承認で Gemini に回る（`_make_minutes_here` の --approved-external）。
            #   承認の文面に書いていないものは送らない、を守るため、ここで並べる
            purpose = "要約と会議のあとの議事録のため" if self._dispatch_cfg.on_finish == "minutes" else "要約のため"
            items.append(f"会議中の全文と事前資料（{purpose}。音声は送りません）")
        if wants_verify:
            items.append("🔍 を押した行と前後の文字（Google 検索で裏取りするため）")
        try:
            approval = approve(config, self.session_dir, [],
                               ask=lambda _prompt: self._ask_external_live(config, items))
        except ExternalSendRefused as refused:
            if wants_stt:
                self._fall_back_to_local(str(refused))
            if wants_llm:
                self._fall_back_llm_to_local(str(refused))
            if wants_verify:
                self._verifier.disable(f"外への送信を承認していません（{refused}）")
            self._publish_models()
            return
        if wants_stt:
            self._start_live_stt(approval)
        if wants_llm:
            self._start_gemini_llm()
        if wants_verify:
            self._start_gemini_verify(approval)
        self._publish_models()

    def _publish_models(self) -> None:
        """画面は起動時に一度だけ /api/status を読む。エンジンが決まるのはそのあとなので、
        ここで知らせないと「外で回しているのに画面はローカルのまま」になる（2026-09-14 に指摘）。"""
        self.bus.publish("status", {"phase": "running", "warnings": [],
                                    "models": self.session_info()["models"]})

    def _start_live_stt(self, approval) -> bool:
        """30 秒刻みで外へ投げる口を開く。承認は `_start_external_engines` で取り済み。

        送り先は `meeting.external_stt.provider`（gemini | deepgram）。どちらでも同じ関門を通る。
        """
        config = self._external_cfg
        try:
            if str(config.provider).lower() == "deepgram":
                from src.stt.deepgram_transcribe import DeepgramApi, DeepgramConfig

                api = DeepgramApi(load_key(config.deepgram_key_file), DeepgramConfig(
                    model=config.deepgram_model, timeout_sec=int(config.timeout_sec)))
            else:
                api = TranscribeApi(load_key(config.key_file), TranscribeConfig(
                    model=config.model, timeout_sec=config.timeout_sec))
        except (OSError, ValueError) as error:
            return self._fall_back_to_local(f"API キーを読めません（{error}）")

        # 刻み方はエンジンで変える（Deepgram は往復 1.5 秒なので 30 秒も溜める必要がない）。
        #   設定に書いてある値が勝つ。`auto` と空は「エンジンに合わせる」の意味。
        #   解決は `src/stt/live_batch.config_from` に 1 つだけ置く（ここに書くと、
        #     同じことをする `scripts/selftest_live.py` とずれる。2026-09-20 に実際にずれた）。
        live_cfg = live_config_from(config.provider, config.live)
        logger.info("会議中の刻み: %s は %.0f 秒ごと（最初だけ %.0f 秒）",
                    config.provider, live_cfg.window_sec, live_cfg.first_window_sec)
        self._live_stt = LiveBatchStt(
            api,
            self.session_dir / "external",
            self._on_live_rows,
            config=live_cfg,
            sample_rate=self._audio_cfg.sample_rate,
            keys=(SELF_KEY, REMOTE_KEY),
            on_gap=self._on_live_gap,
        )
        engine_name = (config.deepgram_model if str(config.provider).lower() == "deepgram"
                       else config.model)
        # 画面の「モデル」欄に出す。送り先を取り違えて見せない（2026-09-20 まで
        #   Deepgram で回していても「外（Gemini・30 秒刻み）」と出ていた）。
        self._live_stt_label = {
            "name": engine_name,
            "where": f"外（{'Deepgram' if str(config.provider).lower() == 'deepgram' else 'Gemini'}"
                     f"・{live_cfg.window_sec:.0f} 秒刻み）",
        }
        record_send(self.session_dir, approval, model=engine_name, note="live_batch")
        # 外のエンジンには Whisper 用の辞書がほとんど効かない（2026-09-13 実測: 230 語中 4 箇所）。
        #   会議中の置き換えもエンジンに合わせて差し替える。
        if config.glossary_path:
            path = Path(config.glossary_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            self._glossary = task_hub_dictionary.merge_pairs(load_glossary(path),
                                                          task_hub_dictionary.live_pairs(self._task_hub_pairs))
            self._glossary_protect = load_protected(path)
        logger.info("会議中の文字起こしを外で回します: %s（%.0f 秒刻み）",
                    engine_name, live_cfg.window_sec)
        print(f"\n  会議中の文字起こしは外（{engine_name}）で回します。"
              f"最初の 1 行は約 {live_cfg.first_window_sec + 13:.0f} 秒後、"
              f"以後は {live_cfg.window_sec:.0f} 秒ごとにまとめて出ます"
              f"（平均 {live_cfg.window_sec / 2 + 13:.0f} 秒・最大 {live_cfg.window_sec + 13:.0f} 秒の遅れ）。")
        return True

    def _fall_back_to_local(self, reason: str) -> bool:
        """外へ出さずに手元の Whisper で起こす。会議が失われてはいけない。"""
        self._stt_engine = "local"
        self._live_stt = None
        self._live_stt_label = {}
        logger.warning("会議中の文字起こしは手元で行います: %s", reason)
        print(f"\n  外へは出しません（{reason}）。会議中の文字起こしは手元で行います。")
        self.bus.publish("status", {"phase": "running",
                                    "warnings": [f"外へは出しません（{reason}）。手元で文字起こしします"]})
        logger.info("Whisper モデルを読み込んでいます…")
        self.whisper._ensure_model()
        return False

    def _start_gemini_llm(self) -> bool:
        """会議中の要約を外（Gemini）で回す。送るのはテキストだけで、音声は送らない。

        速さのためではない（単体ならローカルでも 1 回 3.5 秒）。狙いは
        ①決定事項の拾いの差（同じ入力で 29 件 対 2 件・2026-09-13 実測）と
        ②マシンスペック依存を下げること（gemma4 は 9.6GB・強い GPU が要る）。
        """
        if self.meeting.summary_mode != "rolling":
            return self._fall_back_llm_to_local(
                "従来の 3 分要約は外では回せません（meeting.summary_mode: rolling にしてください）")
        known = set(GeminiLlmConfig.__dataclass_fields__)
        try:
            client = GeminiClient(GeminiLlmConfig(**{k: v for k, v in self._llm_gemini.items() if k in known}))
        except (ValueError, OSError) as error:
            return self._fall_back_llm_to_local(f"Gemini の用意ができません（{error}）")
        if not client.health_check():
            client.close()
            return self._fall_back_llm_to_local("Gemini に繋がりません（キーかモデル名を確認）")
        previous, self.llm = self.llm, client
        previous.close()
        self._redigest_prep()
        logger.info("会議中の要約を外で回します: %s", client.config.model)
        print(f"  会議中の要約は外（{client.config.model}）で回します。送るのは文字だけです。")
        return True

    def _redigest_prep(self, *, wait: bool = False) -> None:
        """要約を外で回し始めたら、長い資料を**外のモデルでまとめ直す**。

        会議の前（承認の前）に足した資料は、手元の Ollama でまとめられる（読めるのは先頭 3 万字まで）。
        Ollama が止まっていればまとめに失敗し、先頭 2,000 字だけが要約に使われていた（2026-09-15 に気づいた）。
        会議を止めないように裏で回す。
        """
        text = self._prep.text
        if not text or text.startswith("（前回タスク") or len(text) <= self._state_cfg.prior_tasks_chars:
            return
        worker = threading.Thread(
            target=lambda: self._prep.reload(note="要約を外（Gemini）で回すので、資料をまとめ直しました"),
            daemon=True, name="prep-redigest")
        worker.start()
        if wait:
            worker.join()

    def _fall_back_llm_to_local(self, reason: str) -> bool:
        """要約は手元の Ollama で回す。Ollama も居なければ、要約なしで会議は続ける。"""
        self._llm_engine = "local"
        logger.warning("会議中の要約は手元で行います: %s", reason)
        print(f"  会議中の要約は手元（{self.llm.config.model}）で回します（{reason}）。")
        if not self.llm.health_check():
            print("  ⚠ Ollama に繋がりません。会議中の要約は出ませんが、文字起こしは続きます。")
            logger.error("Ollama に接続できません（要約なしで続行）")
        return False

    def _on_live_rows(self, key: str, window: AudioWindow, rows: list[dict]) -> None:
        """外から返ってきた 1 窓ぶんを、話者を当ててから画面と議事録へ流す。

        話者は外に任せない（Gemini 76% 対 手元 97%）。返ってきた区間の時刻で録音を切り、
        手元の声紋で当てる（`src/audio/segment_labeler.py`）。
        """
        if not rows:
            return
        if key == REMOTE_KEY and self.diarizer is not None:
            known_unknowns = set(self.diarizer.unknown_names)
            dissolved_before = set(self.diarizer.dissolved_names)
            try:
                label_segments(rows, None, window.sample_rate, self.diarizer,
                               slicer=window.slice, on_voice=self._remember_row_voice)
            except Exception:
                logger.exception("話者判定エラー — ラベルなしで続行")
                for row in rows:
                    row["speaker"] = "不明話者?"
            for name in self.diarizer.unknown_names:
                if name not in known_unknowns:
                    logger.info("新しい話者を検出: %s", name)
            for name in self.diarizer.dissolved_names:
                if name not in dissolved_before:
                    self.bus.publish("dissolve", {"name": name})
                    self.bus.publish("participants", {"items": self.participants()})
        else:
            for row in rows:
                row["speaker"] = self.meeting.self_name

        stamp = datetime.now().isoformat()
        self._emit_segments([
            TranscriptSegment(speaker=row["speaker"], text=row["text"],
                              start_time=row["start_time"], end_time=row["end_time"], timestamp=stamp)
            for row in rows
        ])

    def _stitch_with_previous(self, text: str) -> str:
        """窓の切れ目で前の文から落ちた句点を、前の行へ返す。

        2026-09-18 の実会議（Deepgram）で出た形:
        「自分のマイクです」→「。こんにちは、お世話になります。」
        外のエンジンは窓ごとに独立して句読点を付けるので、こちらで戻す。

        直すのは**画面と要約に渡るぶん**（`transcripts.jsonl` は届いたままを残す）。
        議事録の元になる全文は、会議のあとに録音からまるごと作り直すので窓の切れ目が無い。
        """
        previous = getattr(self, "_last_segment", None)
        if previous is None:
            return text
        fixed_previous, fixed_current = stitch(previous.text, text)
        if fixed_previous == previous.text:
            return text
        previous.text = fixed_previous      # buffer と同じ実体なので要約にも効く
        self.bus.publish("amend", {"start_time": previous.start_time, "text": fixed_previous})
        return fixed_current

    def _remember_row_voice(self, row: dict, embedding: np.ndarray) -> None:
        """外から返ってきた行の声を覚える（画面で名前を直したときに過去へ反映するため）。"""
        self._remember_voice(
            TranscriptSegment(speaker=row["speaker"], text=str(row.get("text", "")),
                              start_time=float(row["start_time"]), end_time=float(row["end_time"]),
                              timestamp=""),
            embedding,
        )

    def _on_live_gap(self, key: str, window: AudioWindow, error: Exception) -> None:
        """送れなかった窓を記録する。会議のあとに、その区間だけ拾い直す材料になる。"""
        entry = {
            "at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
            "stream": key,
            "start_time": round(window.start_time, 2),
            "end_time": round(window.end_time, 2),
            "speech_sec": round(window.speech_sec, 2),
            "error": str(error)[:200],
        }
        try:
            with (self.session_dir / GAPS_FILE).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("取りこぼしの記録に失敗")
        logger.warning("外へ送れなかった区間: %s %.0f〜%.0f 秒（%s）",
                       key, window.start_time, window.end_time, error)
        self.bus.publish("status", {
            "phase": "running",
            "warnings": [f"外へ送れなかった区間があります（{window.start_time:.0f}〜{window.end_time:.0f} 秒）。"
                         "会議のあとに拾い直します"],
        })

    def _submit(self, key: str, chunk: AudioChunk) -> None:
        if self._live_stt is not None:
            # 外で起こす（方式④）。ここでは溜めるだけで、30 秒ぶん揃ったところで送られる。
            #   GPU は使わないので、Ollama の要約が Whisper と取り合わなくなる。
            self._live_stt.feed(key, chunk)
            return
        submitted_at = time.time()
        with self._stt_pending_lock:
            self._stt_pending += 1
        work = self._split_and_transcribe if key == REMOTE_KEY else self._transcribe_self

        def timed(chunk: AudioChunk = chunk) -> list[TranscriptSegment]:
            """1チャンクの実処理時間を残す（遅れの原因を後から切り分けるため）。"""
            started = time.time()
            try:
                return work(chunk)
            finally:
                self._stt_work.append((time.time() - started, chunk.end_time - chunk.start_time))

        future = self._executor.submit(timed)
        future.add_done_callback(lambda done: self._on_transcribed(done, submitted_at))

    def _transcribe_self(self, chunk: AudioChunk) -> list[TranscriptSegment]:
        """自分側の固定話者チャンクを文字起こしする。"""
        segment = self.whisper.transcribe(chunk)
        return [segment] if segment is not None else []

    def _split_and_transcribe(self, chunk: AudioChunk) -> list[TranscriptSegment]:
        """相手側チャンクを話者ターンへ分割してからテキスト化する。"""
        try:
            known_unknowns = set(self.diarizer.unknown_names)
            dissolved_before = set(self.diarizer.dissolved_names)
            turns = split_turns(
                chunk.audio,
                chunk.sample_rate,
                self.diarizer,
                self._turn_cfg,
                chunk.start_time,
            )
            segments: list[TranscriptSegment] = []
            for turn in turns:
                if (
                    turn.speaker.startswith("不明話者")
                    and turn.speaker != "不明話者?"
                    and turn.speaker not in known_unknowns
                ):
                    logger.info("新しい話者を検出: %s", turn.speaker)
                    known_unknowns.add(turn.speaker)
                segment = self.whisper.transcribe(slice_chunk(chunk, turn))
                self.diarizer.report_text(turn.speaker, len(segment.text) if segment else 0)
                if segment is not None:
                    segments.append(segment)
                    if turn.embedding.size:
                        self._remember_voice(segment, turn.embedding)
            for name in self.diarizer.dissolved_names:
                if name not in dissolved_before:
                    self.bus.publish("dissolve", {"name": name})
                    self.bus.publish("participants", {"items": self.participants()})
            return segments
        except Exception:
            logger.exception("話者判定エラー — ラベルなしで続行")
            chunk.speaker = "不明話者?"
            segment = self.whisper.transcribe(chunk)
            return [segment] if segment is not None else []

    def _on_transcribed(self, future, submitted_at: float | None = None) -> None:
        if submitted_at is not None:
            with self._stt_pending_lock:
                self._stt_pending -= 1
            self._stt_latencies.append(time.time() - submitted_at)
        try:
            segments = future.result()
        except Exception:
            logger.exception("STT エラー")
            return
        self._emit_segments(segments)

    def _emit_segments(self, segments: list[TranscriptSegment]) -> None:
        """文字起こしの結果を、置き換え辞書 → 画面 → ファイル → 要約へ流す。

        手元（Whisper）でも外（Gemini の 30 秒刻み）でも、出口はここ 1 本にする。
        """
        with self._emit_lock:
            self._emit_locked(segments)

    def _emit_locked(self, segments: list[TranscriptSegment]) -> None:
        base = self._capture_started_at
        for segment in segments:
            segment.text = apply_glossary(segment.text, self._glossary, self._protected_words())
            segment.text = self._stitch_with_previous(segment.text)
            if not segment.text:
                continue                    # 記号だけの行になった（前の行へ返し切った）
            if base is not None:
                # 記録に残す時刻は「話された時刻」。届いた時刻だと、30 秒の窓ぶんが同じ時刻になる
                segment.timestamp = datetime.fromtimestamp(base + segment.start_time).astimezone().isoformat()
            key = round(segment.start_time, 2)
            labels = getattr(self, "_row_labels", {})
            segment.speaker = labels.get(key, self._resolve_alias(segment.speaker))
            with self._buffer_lock:
                self._transcript_buffer.append(segment)

            self.writer.append_transcript(json.dumps(segment.to_dict(), ensure_ascii=False))
            self.bus.publish("segment", segment.to_dict())
            stats = self._speaker_stats.setdefault(segment.speaker, [0, 0])
            stats[0] += 1
            stats[1] += len(segment.text)

            if self._state_loop is not None:
                self._state_loop.push(segment)
            if self._factcheck_loop is not None:
                self._factcheck_loop.push(segment)

            self._last_segment = segment
            mark = "👤" if segment.speaker == self.meeting.self_name else "🎤"
            logger.info("%s %s [%.1fs]: %s", mark, segment.speaker, segment.start_time, segment.text[:80])
        if segments:
            self.bus.publish("participants", {"items": self.participants()})

    def _build_initial_prompt(self) -> str | None:
        """事前資料の用語節から Whisper の固有名詞ヒントを作る。

        登録話者名は入れない。2026-09-11 本番で「参加者A、参加者B」を渡したら、文中に「GX」が 118 回
        紛れ込み（MacWhisper は同じ区間で 2 回）、無音区間ではヒントそのものを読み上げた。
        同じ 160 行を起こし直した比較: ヒントあり 食い違い 119.7%・GX 510 回 / ヒントなし 59.8%・0 回。
        話者名は画面のラベルで、発話の語彙ではない。読みにくい名前は事前資料の「## 用語」に書く。
        """
        terms: list[str] = []
        in_terms = False
        for line in self._prior_tasks.splitlines():
            if line.startswith("## "):
                in_terms = line.strip() == "## 用語"
                continue
            if in_terms and line.strip():
                terms.append(line.strip().lstrip("-・* ").strip())
        prompt = "、".join(value for value in terms if value)
        return prompt[:120] or None

    def rename_speaker(self, old: str, new: str) -> None:
        """UI からの話者名変更を diarizer へ委譲する。"""
        if self.diarizer is None:
            raise ValueError("話者登録はまだ開始されていません")
        known = set(getattr(self.diarizer, "speaker_names", []) or [])
        if not known:
            known.update(getattr(self.diarizer, "enrolled_names", []) or [])
            known.update(getattr(self.diarizer, "unknown_names", []) or [])
        if old in known:
            try:
                self.diarizer.rename(old, new)
                self.diarizer.mark_named(new)
            except KeyError:
                logger.info("Diarizer speaker already absent during rename: %s", old)
        else:
            # 声紋が知らない名前だった（「判定できなかったときの置き名」に付けた場合）。
            #   その名前で出ている行の声を錨にして、**声と名前を結び付ける**。
            #   これをしないと、画面の表示だけが変わり、作り直すと名前が消える（2026-09-14 の会議）。
            self._anchor_from_rows(old, new)
        self._record_rename(old, new)
        self._aliases[old] = new
        self._aliases.pop(new, None)   # 逆向きの改名で輪にならないように
        labels = getattr(self, "_row_labels", None)
        if labels is None:
            self._row_labels = {}
            labels = self._row_labels
        for key, name in labels.items():
            if name == old:
                labels[key] = new
        if old in self._speaker_stats:
            stats = self._speaker_stats.pop(old)
            target = self._speaker_stats.setdefault(new, [0, 0])
            target[0] += stats[0]
            target[1] += stats[1]
        logger.info("Speaker renamed from %s to %s", old, new)
        self._assign_matching_rows(new)

    def _anchor_from_rows(self, old: str, new: str) -> int:
        """いま `old` と表示されている行の声を、`new` の錨にする（何件足したかを返す）。

        短い行は使わない（`anchor_min_sec`）。声の平均ではなく、長い行から数件だけを足す
        — 平均にすると、取り違えた行が混ざったときに全部が引きずられる。
        """
        if self.diarizer is None:
            return 0
        with self._rows_lock:
            rows = [row for row in self._row_voices.values()
                    if self._current_label(row) == old
                    and row.end_time - row.start_time >= self.meeting.anchor_min_sec]
        rows.sort(key=lambda row: row.end_time - row.start_time, reverse=True)
        added = 0
        for row in rows[:self.meeting.rename_anchor_rows]:
            try:
                self.diarizer.add_anchor(new, row.embedding)
                added += 1
            except ValueError:
                continue
        if added:
            self.diarizer.mark_named(new)
            logger.info("「%s」の声を %d 件覚えました（%s と表示されていた行から）", new, added, old)
        else:
            logger.warning("「%s」に結び付けられる声がありませんでした（作り直しで名前が消えます）", new)
        return added

    def _record_rename(self, old: str, new: str) -> None:
        """改名をセッションに残す（会議のあとの作り直しで当て直すため）。"""
        entry = {"at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
                 "old": old, "new": new}
        try:
            with (self.session_dir / RENAMES_FILE).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("改名の記録に失敗")

    def _resolve_alias(self, name: str) -> str:
        """改名の連鎖（自分→MYSELF→自分）を辿って現在の表示名を返す。"""
        seen: set[str] = set()
        while name in self._aliases and name not in seen:
            seen.add(name)
            name = self._aliases[name]
        return name

    def participants(self) -> list[dict]:
        """登録済み・検出済み話者と発話統計を返す。"""
        enrolled = self.diarizer.enrolled_names if self.diarizer is not None else []
        unknown = self.diarizer.unknown_names if self.diarizer is not None else []
        names = list(dict.fromkeys([*enrolled, *unknown, *self._speaker_stats]))
        known = set(getattr(self.diarizer, "speaker_names", []) or []) if self.diarizer is not None else set()
        items = [
            {
                "name": name,
                "enrolled": name in enrolled,
                "utterances": self._speaker_stats.get(name, [0, 0])[0],
                "chars": self._speaker_stats.get(name, [0, 0])[1],
                "suggestions": self.voice_suggestions(name) if name not in enrolled else [],
            }
            for name in names
            if name != "不明話者?" and (self._speaker_stats.get(name, [0, 0]) != [0, 0] or name in known)
        ]
        items.append({"name": "不明話者?", "enrolled": False, "utterances": self._speaker_stats.get("不明話者?", [0, 0])[0], "chars": self._speaker_stats.get("不明話者?", [0, 0])[1]})
        return items

    def relabel_segment(self, start_time: float, end_time: float, old: str, new: str) -> None:
        """一発話だけの話者訂正を保存し、その声を覚えて、過去の「不明話者?」も付け直す。

        以前は 1 行のラベルを変えるだけで声を覚えなかったので、直した直後の発話からまた
        不明話者に戻っていた（2026-09-11 本番）。
        """
        self._apply_relabel(start_time, end_time, old, new)
        row = self._row_voice(start_time)
        if row is None:
            return
        row.manual = True   # 人が直した行は、あとの自動付け直しで上書きしない
        if self.diarizer is None or new in {"不明話者?", self.meeting.self_name} or new.startswith("不明話者"):
            return
        if row.end_time - row.start_time < self.meeting.anchor_min_sec:
            logger.info("短い行（%.1f 秒）なので声は覚えません: %s", row.end_time - row.start_time, new)
            return
        self.diarizer.add_anchor(new, row.embedding)
        self._assign_matching_rows(new)

    def _remember_voice(self, segment: TranscriptSegment, embedding: np.ndarray) -> None:
        with self._rows_lock:
            self._row_voices[round(segment.start_time, 2)] = _RowVoice(
                start_time=segment.start_time,
                end_time=segment.end_time,
                embedding=np.asarray(embedding, dtype=np.float32),
                speaker=segment.speaker,
            )
            while len(self._row_voices) > self.meeting.max_row_voices:
                self._row_voices.pop(next(iter(self._row_voices)))

    def _row_voice(self, start_time: float) -> _RowVoice | None:
        with self._rows_lock:
            return self._row_voices.get(round(start_time, 2))

    def _current_label(self, row: _RowVoice) -> str:
        labels = getattr(self, "_row_labels", {})
        return labels.get(round(row.start_time, 2), self._resolve_alias(row.speaker))

    def _assign_matching_rows(self, name: str) -> list[float]:
        """いま「不明話者?」の行のうち、声が `name` にはっきり近いものを付け直す。"""
        if self.diarizer is None:
            return []
        cfg = self.diarizer.config
        with self._rows_lock:
            rows = [row for row in self._row_voices.values() if not row.manual]
        changed: list[_RowVoice] = []
        for row in rows:
            if self._current_label(row) != "不明話者?":
                continue
            ordered = sorted(self.diarizer.scores(row.embedding).items(), key=lambda item: item[1], reverse=True)
            if not ordered or ordered[0][0] != name:
                continue
            margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else float("inf")
            # 途中の短い発話と同じ基準（床と、2 位との差）で付ける
            if ordered[0][1] >= cfg.short_turn_threshold and margin >= cfg.short_turn_margin:
                changed.append(row)
        for row in changed:
            self._apply_relabel(row.start_time, row.end_time, "不明話者?", name, auto=True)
            self.bus.publish("relabel", {"start_time": row.start_time, "end_time": row.end_time, "old": "不明話者?", "new": name})
        if changed:
            logger.info("声が %s に近い過去の行 %d 件を付け直しました", name, len(changed))
            self.bus.publish("participants", {"items": self.participants()})
        return [row.start_time for row in changed]

    def _apply_relabel(self, start_time: float, end_time: float, old: str, new: str, auto: bool = False) -> None:
        """一発話の話者訂正を記録し、表示ラベル・未処理バッファ・参加者の集計に反映する。"""
        record = {
            "at": datetime.now().astimezone().isoformat(),
            "start_time": start_time,
            "end_time": end_time,
            "old": old,
            "new": new,
        }
        if auto:
            record["auto"] = True
        self.writer.append_correction(record)
        labels = getattr(self, "_row_labels", None)
        if labels is None:
            self._row_labels = {}
            labels = self._row_labels
        labels[round(start_time, 2)] = new
        chars = 0
        with self._buffer_lock:
            for segment in self._transcript_buffer:
                if abs(segment.start_time - start_time) <= 0.05 and segment.speaker == old:
                    segment.speaker = new
                    chars = len(segment.text)
        if chars == 0:
            chars = next(
                (len(item.text) for item in self._read_transcripts() if abs(item.start_time - start_time) <= 0.05),
                0,
            )
        # 参加者リストの発話数・文字数も 1 発話ぶん移す（移さないと改名前後が両方残る）
        if old in self._speaker_stats:
            self._speaker_stats[old][0] = max(0, self._speaker_stats[old][0] - 1)
            self._speaker_stats[old][1] = max(0, self._speaker_stats[old][1] - chars)
            known = set(getattr(self.diarizer, "speaker_names", []) or []) if self.diarizer is not None else set()
            if self._speaker_stats[old] == [0, 0] and old not in known and old != "不明話者?":
                self._speaker_stats.pop(old)
        target = self._speaker_stats.setdefault(new, [0, 0])
        target[0] += 1
        target[1] += chars
        logger.info("Segment relabeled from %s to %s at %.3f", old, new, start_time)

    def session_info(self) -> dict:
        """UI の初期表示に必要なセッション情報を返す。"""
        return {
            "session_name": self.session_dir.name,
            # 字幕の時刻は「開始 + 経過」で出すので、開始は**録音の先頭**でなければならない
            "started_at": datetime.fromtimestamp(
                self._capture_started_at or self._started_at).astimezone().isoformat(),
            "self_name": self.meeting.self_name,
            "enrolled": self.diarizer.enrolled_names if self.diarizer is not None else [],
            # 画面が「次の更新まで」を出すのに使う。**画面に数字を直書きしない**——
            #   09-18 に 30 秒 → 2 分へ変えたのに、画面は「30 秒以内」と言い続けていた（09-21 に発見）
            "state_interval_sec": float(self._state_cfg.interval_sec),
            "dispatch": {
                "enabled": self._dispatch_cfg.enabled,
                "agent_repo": bool(self._dispatch_cfg.agent_repo),
                "slack": bool(self._bridge is not None and self._task_hub_cfg.slack),
                "notion_tasks": bool(self._bridge is not None and self._task_hub_cfg.notion_tasks),
            },
            "models": {
                "stt": (dict(self._live_stt_label)
                        if self._live_stt is not None and self._live_stt_label
                        else {"name": self.whisper.config.model, "where": "ローカル（mlx）"}),
                "summary": {"name": self.llm.config.model,
                            "where": "外（Gemini）" if self._llm_engine == "gemini" else "ローカル（Ollama）",
                            "think": self._state_cfg.think},
                "final_pass": {"name": "Claude CLI", "where": "外部（テキストのみ・音声は出ない）"} if self._final_pass != "none" else None,
                "verify": self._verifier.model_info(),
            },
        }

    # ------------------------------------------- 画面から頼む裏取り（Web 検索つき）
    #
    # 中身は `src/meeting/verify.py`。ここは画面から呼ばれる口と、文字起こしの引き当てだけ。

    def verify_row(self, start_time: float, text: str = "") -> dict:
        """画面で指された 1 行を、Web で裏取りする（裏で走る。結果は画面へ流れる）。"""
        return self._verifier.request(start_time, text)

    def _text_at(self, start_time: float) -> str:
        """その時刻の行の本文を、会議中のバッファか書き出し済みの全文から拾う。"""
        with self._buffer_lock:
            for segment in self._transcript_buffer:
                if abs(segment.start_time - start_time) <= 0.05:
                    return segment.text
        return next((item.text for item in self._read_transcripts()
                     if abs(item.start_time - start_time) <= 0.05), "")

    def _context_around(self, start_time: float) -> str:
        """前後 `VERIFY_CONTEXT_SEC` 秒の文字起こし（何の話かを分かるようにするためだけ）。"""
        rows = [item for item in self._read_transcripts()
                if abs(item.start_time - start_time) <= VERIFY_CONTEXT_SEC]
        return self._format_transcript(rows)

    def dispatch(self, todo_id: str) -> dict:
        """UI 指定の TODO をブリーフと各連携先へ払い出す。"""
        if self._state_loop is None:
            raise ValueError("会議状態はまだ開始されていません")
        state = self._state_loop.state()
        todo = next((item for item in state.todos if item.id == todo_id), None)
        if todo is None:
            raise KeyError(todo_id)
        owner = todo.by if todo.by and todo.by != "未定" else self.meeting.self_name
        session = {
            "client_name": self._client_name(),
            "meeting_date": datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat(),
            "session_name": self.session_dir.name,
        }
        result = dispatch(
            self.session_dir,
            state,
            todo,
            self._read_transcripts(),
            session,
            self._dispatch_cfg,
            self._bridge,
            self._agent,
            owner,
        )
        self._dispatched.append(result)
        return self._dispatch_result_dict(result)

    def _read_transcripts(self) -> list[TranscriptSegment]:
        """永続化済み transcripts.jsonl を時系列セグメントとして読む。"""
        path = self.session_dir / "transcripts.jsonl"
        if not path.exists():
            return []
        segments: list[TranscriptSegment] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                segments.append(TranscriptSegment(**json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("文字起こし行を読み飛ばしました: %s", exc)
        # 書かれた順は時刻順とは限らない（2 系統を同時に起こすので前後する）。議事録の
        #   全文はこの並びのまま出るので、ここで揃える。
        return sorted(segments, key=lambda segment: segment.start_time)

    @staticmethod
    def _dispatch_result_dict(result: DispatchResult) -> dict:
        """Path を含む払い出し結果を JSON 応答用の辞書に変換する。"""
        return {
            "todo_id": result.todo_id,
            "brief_path": str(result.brief_path),
            "already": result.already,
            "notion": result.notion,
            "slack_ok": result.slack_ok,
            "agent": result.agent,
            "errors": result.errors,
        }

    # ------------------------------------------------------------ 要約

    def _on_state(self, state: MeetingState, stats: UpdateStats, changes: list[str]) -> None:
        """ローリング状態を保存し、変更行をタイムラインへ追記する。"""
        try:
            self.writer.write_state(state)
            self.writer.write_state_json(state)
            self._last_state_latency_sec = stats.latency_sec
            self.bus.publish("state", {"state": state.to_dict(), "latency_sec": stats.latency_sec, "parse_ok": stats.parse_ok})
            if changes:
                at = self._format_seconds(state.updated_at)
                self.writer.append_log("\n".join(changes), start_time=at, end_time=at)
            logger.info(
                "状態更新: prompt=%d chars, %.1fs, parse_ok=%s, changes=%d",
                stats.prompt_chars,
                stats.latency_sec,
                stats.parse_ok,
                len(changes),
            )
            if not stats.parse_ok:
                logger.warning("状態更新の JSON 解析に失敗しました")
        except Exception:
            logger.exception("状態更新の出力に失敗")

    def _run_summary_cycle(self) -> None:
        with self._buffer_lock:
            buffer_copy = list(self._transcript_buffer)
            self._transcript_buffer.clear()

        recent_text = self._format_transcript(buffer_copy)

        try:
            response = self.llm.summarize(
                system_prompt=self._system_prompt,
                hearing_items=self._prior_tasks,
                overall_summary_so_far=self._overall_summary,
                recent_transcript=recent_text,
            )
        except Exception:
            logger.exception("Ollama 要約エラー — バッファを次サイクルに繰り越し")
            with self._buffer_lock:
                self._transcript_buffer = buffer_copy + self._transcript_buffer
            return

        if response.overall_summary:
            self._overall_summary = response.overall_summary

        self.writer.overwrite_summary(response)

        start = self._format_seconds(buffer_copy[0].start_time) if buffer_copy else ""
        end = self._format_seconds(buffer_copy[-1].end_time) if buffer_copy else ""
        self.writer.append_log(response.chunk_summary, start_time=start, end_time=end)

        logger.info("=== 要約サイクル完了 ===")

    def _format_transcript(self, segments: list[TranscriptSegment]) -> str:
        """セグメントを時系列テキストへ。話者名はそのまま出す（登録名／不明話者N）。"""
        lines = []
        for seg in sorted(segments, key=lambda s: s.start_time):
            lines.append(f"[{self._format_seconds(seg.start_time)}] {seg.speaker}: {seg.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

    def _extract_purpose(self) -> str:
        for line in self._prior_tasks.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
        return ""

    # ------------------------------------------------------------ 終了

    def _shutdown(self, speakers_path: Path) -> None:
        logger.info("シャットダウン処理中…")
        self.bus.publish("status", {"phase": "stopping", "warnings": []})
        # 画面はここでは閉じない。会議のあとの作り直しで「外へ出すか」を画面で聞くため
        #   （運用者 依頼 2026-09-13「実会議でやるなら GUI で見たい」）。閉じるのはこの関数の最後。
        if self.capture is not None:
            self.capture.stop()
        unfinished = self._verifier.unfinished()
        if unfinished:
            # 裏取りは daemon スレッドなので、ここで終わると黙って消える。何が消えたかを残す
            logger.warning("裏取りの途中で会議が終わりました: %s 秒の行 %d 件",
                           "・".join(f"{value:.0f}" for value in unfinished), len(unfinished))
            print(f"\n  ※ 裏取りが {len(unfinished)} 件、途中で終わりました"
                  f"（{self.session_dir / VERIFY_FILE} には残りません）。必要なら会議のあとに調べ直してください。")
        if self._screens is not None:
            self._screens.stop()
            logger.info("画面共有の取り込み: %d 枚（変わらず飛ばしたのは %d 回）",
                        self._screens.shots, self._screens.skipped)
            if self._screens.shots:
                print(f"  画面共有を {self._screens.shots} 枚控えました（{self.session_dir / 'screens'}）")
        if self._live_stt is not None:
            # 溜まっている最後の窓を送り、返ってくるのを待つ（待たないと会議の終わりが丸ごと消える）
            print("\n  外へ送った最後のぶんを待っています…")
            self._live_stt.flush()
            self._live_stt.close(wait=True)
            usage = self._live_stt.usage
            logger.info("会議中の外部文字起こし: %d 窓 / 入力 %d・出力 %d トークン / "
                        "失敗 %d・拾い直し待ち %d / 往復の中央 %.1f 秒",
                        usage.windows, usage.prompt_tokens, usage.output_tokens,
                        usage.failures, usage.gaps, usage.median_latency_sec)
            if usage.gaps:
                print(f"  ⚠ 外へ送れなかった区間が {usage.gaps} 件あります"
                      f"（{self.session_dir / GAPS_FILE}）。作り直しで拾い直します。")
        self._executor.shutdown(wait=True, cancel_futures=False)

        # 実行中に増えた暫定話者も残す（次回の名寄せ材料になる）
        if self.diarizer is not None and len(self.diarizer):
            try:
                self.diarizer.save(speakers_path)
            except Exception:
                logger.exception("話者登録の保存に失敗")

        if self.meeting.summary_mode == "rolling" and self._state_loop is not None:
            self._state_loop.stop()
            state = self._state_loop.flush(timeout=60)
            self.writer.write_state(state)
            self.writer.write_state_json(state)
        else:
            with self._buffer_lock:
                remaining = list(self._transcript_buffer)
                self._transcript_buffer.clear()

            if remaining:
                logger.info("残りバッファ %d セグメントの最終要約を実行", len(remaining))
                try:
                    response = self.llm.summarize(
                        system_prompt=self._system_prompt,
                        hearing_items=self._prior_tasks,
                        overall_summary_so_far=self._overall_summary,
                        recent_transcript=self._format_transcript(remaining),
                    )
                    self.writer.overwrite_summary(response)
                    self.writer.append_log(response.chunk_summary)
                except Exception:
                    logger.exception("最終要約に失敗")

            state = MeetingState()

        if self._final_pass == "claude_cli":
            client = ClaudeCliClient()
            if client.available():
                try:
                    delta = client.final_pass(state, self._format_transcript(self._read_transcripts()), final_pass_prompt())
                    state, _ = apply_delta(state, delta, max(state.updated_at, 0.0))
                    self.writer.write_state(state)
                    self.writer.write_state_json(state)
                except Exception:
                    logger.warning("Claude CLI 最終パスに失敗しました", exc_info=True)

        self._confirm_client()      # 引き渡す前に聞く（初回面談がその場で案件になることがある）
        corrections = self._read_corrections()
        session = {
            "client_name": self._client_name(),
            "meeting_date": datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat(),
            "session_name": self.session_dir.name,
        }
        _, handoff = write_minutes_handoff(
            self.session_dir,
            state,
            self._read_transcripts(),
            session,
            self._dispatched,
            corrections,
            mark_aizuchi=self.meeting.aizuchi_mark,
        )
        self.llm.close()
        if self._verify_gemini is not None:
            self._verify_gemini.close()

        if self.diarizer is not None and self.diarizer.unknown_names:
            print()
            print("※ 名前の付かなかった話者がいます:", "、".join(self.diarizer.unknown_names))
            print(f"  {speakers_path} に残してあるので、後から名寄せできます。")

        if self.diarizer is not None:
            self._warn_failed_enrollments(speakers_path)

        try:
            if not self._ask_finish():
                self._stop_here(handoff)
                return
            if self.meeting.finalize_after:
                self._finalize_from_recording()
            self._learn_voices()
            self._sync_task_hub_dictionary()
            self._suggest_glossary()
            # 議事録へ渡すのは**作り直しが終わってから**。前は作り直しより先に議事録の机を開いていたので、
            #   minutes_input.md を書き換えている最中（約 100 秒）に議事録化が読み始めていた。
            self._hand_over_minutes(handoff)
            self._show_outputs(handoff)
        finally:
            # 何が起きても画面は閉じる（daemon スレッドだが、閉じ忘れるとポートが残る）
            if self._ui_thread is not None:
                self._ui_thread.server.should_exit = True
        logger.info("シャットダウン完了")

    # --------------------------------------- 仕上げ（押してから走らせる・あとで再開できる）

    def _ask_finish(self) -> bool:
        """このまま仕上げてよいか聞く。返事が無ければ**後回し**（勝手に数分の処理を始めない）。"""
        mode = str(self.meeting.finish_mode).lower()
        if mode == "auto":
            return True
        wanted = ["finalize"] if self.meeting.finalize_after else []
        wanted += ["voices", "glossary", "minutes"]
        steps = finish.remaining(self.session_dir, wanted=wanted,
                                  voice_enabled=self._voice_cfg.enabled)
        if not steps:
            return True
        payload = {"ask": True, "saved": finish.saved_already(self.session_dir),
                   "steps": [{"key": s.key, "label": s.label, "minutes": s.minutes} for s in steps],
                   "timeout_sec": self.meeting.finish_wait_sec, "session": self.session_dir.name}
        if mode == "later":
            return False
        if self._ui_thread is None:
            return True             # 画面が無いときは従来どおり（端末で見ている＝待てる状況）
        self._finish_decided.clear()
        self._finish_choice = ""
        self.bus.publish("finish", payload)
        print("\n  ── ここまでは残りました ──\n    " + "／".join(payload["saved"]))
        print("  このあとの仕上げ（" + "／".join(f"{s.label}（{s.minutes}）" for s in steps) + "）を、"
              "いま走らせますか?\n    画面の「いま仕上げる」か「あとでやる」を押してください"
              f"（{self.meeting.finish_wait_sec:.0f} 秒で**あとでやる**に倒します）")
        self._finish_decided.wait(self.meeting.finish_wait_sec)
        return self._finish_choice == "now"

    def finish_now(self, choice: str) -> dict:
        """画面の「いま仕上げる／あとでやる」。押されるまで重い処理は始まらない。"""
        self._finish_choice = "now" if str(choice).lower() in {"now", "いま", "yes"} else "later"
        self._finish_decided.set()
        self.bus.publish("finish", {"ask": False, "choice": self._finish_choice})
        return {"ok": True, "choice": self._finish_choice}

    def _stop_here(self, handoff: Path) -> None:
        """仕上げずに終わる。何が残っているかを書いて、再開の仕方を出す。"""
        wanted = ["finalize"] if self.meeting.finalize_after else []
        wanted += ["voices", "glossary", "minutes"]
        steps = finish.remaining(self.session_dir, wanted=wanted,
                                  voice_enabled=self._voice_cfg.enabled)
        finish.write_pending(self.session_dir, steps, note="会議の終わりに「あとでやる」を選んだ")
        print("\n  仕上げは後回しにしました。録音と全文は残っています。")
        print(f"    残り: {'／'.join(step.label for step in steps)}")
        print("    再開: 「会議アシスタント.command」→「会議のあと」で「仕上げる」を押す")
        print(f"      （端末でやるなら: python scripts/finish_meeting.py {self.session_dir}）")
        self._show_outputs(handoff)

    def _hand_over_minutes(self, handoff: Path) -> None:
        """議事録の材料を、設定された次の工程へ渡す（`dispatch.on_finish`）。"""
        on_finish = self._dispatch_cfg.on_finish
        if on_finish == "minutes":
            self._make_minutes_here()
            return
        argv = finish_command(handoff, self._dispatch_cfg) if on_finish in {"agent", "print"} else []
        if on_finish == "agent" and self._agent is not None:
            try:
                completed = subprocess.run(argv, shell=False, check=False)
                if completed.returncode != 0:
                    logger.warning("議事録 Agent 起動に失敗しました: %s", argv)
            except OSError:
                logger.warning("議事録 Agent 起動に失敗しました: %s", argv, exc_info=True)
        elif on_finish == "print":
            print(argv)
        if self._bridge is not None:
            self._bridge.notify(f"■ 会議終了（着手済み {len(self._dispatched)} 件）。議事録処理を開始: {handoff}")

    def _make_minutes_here(self) -> None:
        """議事録（minutes.md）まで、この場で作る（`scripts/make_minutes.py`）。

        使える手段を上から選ぶ: Claude CLI → Gemini → 手元の LLM → 貼り付け用の指示書。
        Gemini は、**この会議で外へのテキスト送信を承認したとき**だけ候補にする。
        「手元で作る」を選んだ会議では、会議のあとで改めて外へ出すかを聞かない。
        """
        script = Path(__file__).resolve().parent.parent / "scripts" / "make_minutes.py"
        if not script.exists():
            return
        argv = [sys.executable, str(script), str(self.session_dir)]
        argv.append("--approved-external" if self._llm_engine == "gemini" else "--no-external")
        print()
        print("=" * 66)
        print("  議事録を作っています（このまま待ってください）")
        print("=" * 66)
        try:
            subprocess.run(argv, check=False)
        except KeyboardInterrupt:
            print("\n  議事録づくりを中断しました（議事録の材料 minutes_input.md は残っています）。")
        except OSError:
            logger.warning("議事録づくりを起動できませんでした", exc_info=True)

    def _wait_for_other_apps(self) -> None:
        """ほかのアプリ（MacWhisper 等）の処理が落ち着くまで待つ。

        MacWhisper は会議が終わると自動で文字起こしを始め、止める設定が無い（2026-09-12 に確認）。
        運用実績があるので外さない（運用者 判断）。代わりに**こちらが待つ**。
        2026-09-11 は取り合いで GPU リセットが 2 回起き、文字起こしと要約が落ちた。
        """
        names = list(self.meeting.wait_for_apps)
        if not names:
            return
        print(f"\n  {'・'.join(names)} の処理が落ち着くのを待っています…")

        def report(cpu: float, waited: float) -> None:
            print(f"    {'・'.join(names)}: CPU {cpu:.0f}%（{waited / 60:.0f} 分待機中）")

        result = wait_until_quiet(
            names,
            threshold_percent=self.meeting.wait_cpu_percent,
            timeout_sec=self.meeting.wait_timeout_sec,
            on_wait=report,
        )
        logger.info("ほかのアプリの待機: %s", result)
        if result["reason"] == "待ち時間の上限":
            print(f"    待ち時間の上限（{self.meeting.wait_timeout_sec / 60:.0f} 分）に達したので先へ進みます。")
        else:
            print(f"    落ち着きました（{result['waited_sec'] / 60:.0f} 分待機）。作り直しを始めます。")

    def _warn_failed_enrollments(self, speakers_path: Path) -> None:
        """声の登録が失敗していたら、終わったときに言う。

        2026-09-11 の本番では、登録した「参加者B」の声紋が本人の声とまったく一致しておらず
        （平均類似度 0.605）、実際は自動で育った `不明話者1` が本人を担っていた。それでも
        画面には何も出ず、失敗したことに気づけたのは 2 日後に採点したときだった。
        議事録では幻の登録話者が相づちを吸って別人として載るので、その場で言う。
        """
        lines = failed_enrollment_lines(self.diarizer, speakers_path)
        if not lines:
            return
        print()
        for line in lines:
            print(line)
        if self._bridge is not None:
            names = "、".join(failure["name"] for failure in self.diarizer.failed_enrollments())
            self._bridge.notify(f"⚠ 声の登録が効いていない可能性: {names}（不明話者が代わりに担当）")

    def _recordings(self) -> list[Path]:
        """外へ出す候補の録音（存在するものだけ）。"""
        return [path for path in (self.session_dir / "recording_remote.wav",
                                  self.session_dir / "recording_self.wav") if path.exists()]

    def _show_outputs(self, handoff: Path) -> None:
        """できたものの置き場を出し、フォルダを開く（パスを辿らなくていいように）。"""
        print()
        print("=" * 66)
        print("  できたもの")
        print("=" * 66)
        for name, label in (("minutes.md", "議事録"),
                            ("minutes_prompt.md", "議事録の指示書（AI チャットに貼り付けると議事録になる）"),
                            ("minutes_input.md", "議事録の材料（これを /task_hub-minutes へ）"),
                            ("interview_summary.md", "会議ダッシュボード（要約・決定事項・TODO）"),
                            ("transcripts_final.jsonl", "全文（録音から作り直した版）"),
                            ("transcripts.jsonl", "全文（会議中のもの）"),
                            ("recording_remote.wav", "録音（相手側）"),
                            ("recording_self.wav", "録音（自分側）")):
            path = self.session_dir / name
            if path.exists():
                print(f"  {label}\n    {path}")
        print(f"\n  引き渡し: {handoff}")
        if self.meeting.open_folder_after and not os.environ.get("MEETING_NO_OPEN"):
            try:
                subprocess.run(["open", str(self.session_dir)], check=False)
            except OSError:
                logger.debug("フォルダを開けませんでした", exc_info=True)

    def _finalize_from_recording(self) -> None:
        """録音から全文を起こし直して、議事録の材料を作り直す。

        会議中は実時間に追いつく必要があるので取りこぼしが出る。終わってしまえば急ぐ理由はない
        （2026-09-11 の会議: 会議中 0.7 倍速 → 会議の外なら 12 倍速。文字の食い違いは 65.8% → 30.3%）。
        49 分の会議で 5〜10 分ほどかかる。切りたいときは settings の `meeting.finalize_after: false`。
        """
        script = Path(__file__).resolve().parent.parent / "scripts" / "finalize_meeting.py"
        if not script.exists():
            return
        self._wait_for_other_apps()
        print()
        print("=" * 66)
        print("  録音から全文を起こし直しています（このまま待ってください）")
        print("  49 分の会議で 5〜10 分ほど。中断しても、会議中の記録は残ります。")
        print("=" * 66)
        argv = [sys.executable, str(script), str(self.session_dir)]
        config = self._external_cfg
        if self._live_stt is not None:
            # 方式④: 会議中に外で起こしたので、**同じ音声を送り直さない**。
            #   話者を当て直し、送れなかった区間だけ拾い直す（そこだけもう一度聞く）。
            argv += ["--reuse-live"]
            gaps = self._gap_entries()
            if gaps and self._ask_external_gaps(config, gaps):
                argv += ["--external-stt", "--approved-in-ui"]
            else:
                argv += ["--no-external-stt"]
        elif config.enabled and config.billing_project:
            if self._ask_external_send(self._recordings(), config):
                # 人が画面（または端末）で答えた直後だけ付く。手で付けるものではない
                argv += ["--external-stt", "--approved-in-ui"]
            else:
                argv += ["--no-external-stt"]
        try:
            subprocess.run(argv, check=False)
        except KeyboardInterrupt:
            print("\n  作り直しを中断しました（会議中の記録はそのまま残っています）。")
        except OSError:
            logger.warning("作り直しを起動できませんでした", exc_info=True)

    def _read_corrections(self) -> list[dict]:
        """保存済みの話者訂正を読み、壊れた行は無視する。"""
        path = self.session_dir / "corrections.jsonl"
        if not path.exists():
            return []
        corrections: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                corrections.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("訂正行を読み飛ばしました: %s", exc)
        return corrections


def _render_prep_digest(data: dict, original_chars: int, limit: int) -> str:
    """まとめを、会議中ずっと載せる短いメモに整える。

    長さは**こちらで詰める**。1,200 文字以内と指示しても、実測（gemma4・2026-09-14）では
    2,800 文字返ってきた。溢れるときは「確認したい論点」の後ろから落とす
    — 前回タスク・数字・用語は照合に使うので残す。
    """
    def bullets(key: str) -> list[str]:
        return [f"- {str(value).strip()}" for value in (data.get(key) or []) if str(value).strip()]

    head = [f"### 事前資料のまとめ（自動・元は {original_chars:,} 文字）"]
    purpose = str(data.get("purpose", "")).strip()
    if purpose:
        head.append(f"目的: {purpose}")
    sections: dict[str, list[str]] = {}
    for key, label in (("tasks", "前回タスク・宿題"), ("points", "確認したい論点"),
                       ("numbers", "数字・期日"), ("terms", "用語")):
        items = bullets(key)
        if items:
            sections[key] = [f"\n**{label}**"] + items

    def rendered() -> str:
        lines = list(head)
        for key in ("tasks", "points", "numbers", "terms"):
            lines += sections.get(key, [])
        return "\n".join(lines)

    while len(rendered()) > limit and len(sections.get("points", [])) > 1:
        sections["points"] = sections["points"][:-1]        # 論点の後ろから落とす
    if len(sections.get("points", [])) <= 1:
        sections.pop("points", None)
    return rendered()[:limit]


def _safe_prep_name(name: str) -> str:
    """画面から来た名前を、セッションの中だけに収まるファイル名にする。"""
    base = Path(str(name)).name.strip()
    base = base.replace("/", "_").replace("\\", "_").lstrip(".")
    if not base:
        raise ValueError("名前がありません")
    return base[:120]


def _normalize(text: str) -> str:
    return "".join(text.lower().split())


@dataclass
class ToneResult:
    """`detect_tone` の結果。"""

    tone_sec: float
    """指定周波数の純音が鳴っていたと判定できた長さ（秒）。"""
    peak_dbfs: float
    dominant_hz: float
    """録音全体でいちばん強い周波数。判定には使わない（ログ用）。"""


def detect_tone(
    audio: np.ndarray,
    sample_rate: int,
    tone_hz: float,
    *,
    frame_sec: float = 0.05,
    min_ratio: float = 0.7,
    min_dbfs: float = -50.0,
) -> ToneResult:
    """録音の中で `tone_hz` の純音が鳴っていた長さを測る。

    「録音全体でいちばん強い周波数が tone_hz か」で判定すると、相手が話している間は
    声（数百 Hz）に負けて必ず落ちる（2026-09-11 実測: 0.2 秒のビープが 264 Hz の声に負けた）。
    そこで 50 ms ごとに区切り、tone_hz 付近（±40 Hz）の成分が周辺帯域（±30%）の大半を
    占めている区間だけを数える。声は倍音が並ぶので周辺帯域を1本で占めることはまず無い。

    限界: 声がビープより 14 dB 以上大きいと埋もれる。逆に倍音が偶然 tone_hz に居座ると真になる。
    どちらも「相手の音が届いている」＝SCK は生きている状況なので、`_preflight` は相手側の
    レベルと合わせて判断する（ビープ単独では警告を出さない）。
    """
    flat = np.asarray(audio, dtype=np.float32)
    if flat.ndim == 2:
        flat = flat.mean(axis=1)
    flat = flat.reshape(-1)
    if flat.size == 0:
        return ToneResult(tone_sec=0.0, peak_dbfs=-100.0, dominant_hz=0.0)

    peak = float(np.max(np.abs(flat)))
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-8))
    spectrum = np.abs(np.fft.rfft(flat))
    dominant_hz = float(np.fft.rfftfreq(flat.size, 1.0 / sample_rate)[int(np.argmax(spectrum))])

    frame_len = int(round(frame_sec * sample_rate))
    if flat.size < frame_len:
        return ToneResult(tone_sec=0.0, peak_dbfs=peak_dbfs, dominant_hz=dominant_hz)
    # 半分ずつ重ねて切る（ビープの端が区間の境目に来ても取りこぼさない）
    hop = frame_len // 2
    frames = np.lib.stride_tricks.sliding_window_view(flat, frame_len)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(frame_len), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)

    # ハン窓の主ローブは ±2 ビン（50 ms なら ±40 Hz）に広がるので、その幅を「純音の帯」にする
    band = np.abs(freqs - tone_hz) <= 2.0 * sample_rate / frame_len
    near = (freqs >= tone_hz * 0.7) & (freqs <= tone_hz * 1.3)
    band_power = power[:, band].sum(axis=1)
    near_power = power[:, near].sum(axis=1)
    total_power = power.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(near_power > 0, band_power / near_power, 0.0)
        # 帯の成分が区間の平均パワーのうちどれだけかを dBFS に直す（ほぼ無音の区間を弾くため）
        frame_ms = np.mean(frames**2, axis=1)
        band_ms = np.where(total_power > 0, band_power / total_power * frame_ms, 0.0)
        band_dbfs = 10.0 * np.log10(np.maximum(band_ms, 1e-12))

    hits = (ratio >= min_ratio) & (band_dbfs >= min_dbfs)
    return ToneResult(tone_sec=float(hits.sum()) * hop / sample_rate, peak_dbfs=peak_dbfs, dominant_hz=dominant_hz)
