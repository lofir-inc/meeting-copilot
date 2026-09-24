"""オーケストレーター — 全コンポーネントの統合・メインループ制御。"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.audio.capture import AudioCapture, AudioConfig
from src.audio.splitter import split_channels
from src.audio.vad import AudioChunk, ChunkBuilder, VadConfig
from src.llm.ollama_client import LlmConfig, OllamaClient
from src.output.markdown_writer import MarkdownWriter, OutputConfig
from src.stt.whisper_client import SttConfig, TranscriptSegment, WhisperClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """インタビュー支援システムのメインオーケストレーター。"""

    def __init__(self, config: dict) -> None:
        # --- Config objects ---
        audio_cfg = AudioConfig(**config.get("audio", {}))
        vad_cfg = VadConfig(**config.get("vad", {}))
        stt_cfg = SttConfig(**config.get("stt", {}))
        llm_cfg = LlmConfig(**config.get("llm", {}))
        output_cfg = OutputConfig(**config.get("output", {}))
        self._summary_interval = config.get("orchestrator", {}).get("summary_interval_sec", 300)

        # --- Components ---
        record_path = Path(output_cfg.workspace_dir) / "recording.wav"
        self.capture = AudioCapture(audio_cfg, record_path=record_path)
        self._session_start = time.time()
        self._swap_lr = False  # キャリブレーションで決定
        self.vad_l = ChunkBuilder("interviewer", vad_cfg, audio_cfg.sample_rate)
        self.vad_r = ChunkBuilder("guest", vad_cfg, audio_cfg.sample_rate)
        self.whisper = WhisperClient(stt_cfg)
        self.ollama = OllamaClient(llm_cfg)
        self.writer = MarkdownWriter(output_cfg)

        # --- State ---
        self._transcript_buffer: list[TranscriptSegment] = []
        self._buffer_lock = threading.Lock()
        self._overall_summary: str = ""
        self._hearing_items: str = ""
        self._system_prompt: str = ""
        self._executor = ThreadPoolExecutor(max_workers=1)  # Metal GPU は同時アクセス不可

    def load_prep_materials(self, prep_dir: str | Path) -> None:
        """事前準備資料（ヒアリング項目等）を読み込む。"""
        prep_path = Path(prep_dir)
        items_parts: list[str] = []

        for md_file in sorted(prep_path.glob("*.md")):
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                items_parts.append(content)
                logger.info("Loaded prep file: %s", md_file.name)

        self._hearing_items = "\n\n".join(items_parts) if items_parts else "（ヒアリング項目未設定）"

        # システムプロンプト読み込み
        prompt_path = Path("prompts/summary_system.md")
        if prompt_path.exists():
            self._system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            logger.warning("System prompt not found: %s", prompt_path)
            self._system_prompt = "あなたはインタビュー支援AIです。会話を要約してください。"

    def run(self) -> None:
        """メインループを開始する。Ctrl+C で停止。"""
        # Ollama ヘルスチェック
        if not self.ollama.health_check():
            logger.error("Ollama に接続できません。ollama serve が起動しているか確認してください。")
            return

        # ワークスペース初期化
        self.writer.interview_purpose = self._extract_purpose()
        self.writer.init_files(self._hearing_items)

        # Whisper モデル事前ロード
        logger.info("Whisper モデルを事前ロード中…")
        self.whisper._ensure_model()
        logger.info("Whisper モデルロード完了。")

        # L/R キャリブレーション
        self._run_calibration()
        logger.info("インタビューを開始できます。")

        # 音声キャプチャ開始（録音込み）
        self.capture.start()
        self._session_start = time.time()
        last_summary_time = time.time()
        segment_count = 0

        try:
            while True:
                # --- 音声処理 ---
                processed = self._process_audio_queue()
                segment_count += processed

                # --- 5分サイクル ---
                elapsed = time.time() - last_summary_time
                if elapsed >= self._summary_interval:
                    with self._buffer_lock:
                        buf_size = len(self._transcript_buffer)
                    if buf_size > 0:
                        logger.info(
                            "=== 要約サイクル開始 (segments=%d, elapsed=%.0fs) ===",
                            buf_size, elapsed,
                        )
                        self._run_summary_cycle()
                    else:
                        logger.info("バッファ空のため要約スキップ")
                    last_summary_time = time.time()

                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Ctrl+C 受信 — シャットダウン中…")
        finally:
            self._shutdown()

    def _run_calibration(self) -> None:
        """L/R キャリブレーション — インタビュアーの声でチャンネル割り当てを決定する。"""
        import sounddevice as sd
        import numpy as np

        print("\n" + "=" * 50)
        print("【キャリブレーション】")
        print("インタビュアーのマイクに向かって数秒話してください。")
        print("3秒間録音します...")
        print("=" * 50 + "\n")

        audio = sd.rec(
            int(3 * self.capture.config.sample_rate),
            samplerate=self.capture.config.sample_rate,
            channels=2,
            dtype="float32",
            device=self.capture.config.device,
        )
        sd.wait()

        l_rms = float(np.sqrt(np.mean(audio[:, 0] ** 2)))
        r_rms = float(np.sqrt(np.mean(audio[:, 1] ** 2)))
        l_db = 20 * np.log10(l_rms) if l_rms > 0 else -100
        r_db = 20 * np.log10(r_rms) if r_rms > 0 else -100

        logger.info("Calibration: L=%.1f dB, R=%.1f dB", l_db, r_db)

        if r_rms > l_rms:
            self._swap_lr = True
            logger.info("L/R スワップ: R ch → Interviewer, L ch → Guest")
            print("→ R チャンネルをインタビュアーに割り当てました\n")
        else:
            self._swap_lr = False
            logger.info("L/R そのまま: L ch → Interviewer, R ch → Guest")
            print("→ L チャンネルをインタビュアーに割り当てました\n")

    def _process_audio_queue(self) -> int:
        """キューから音声を取り出し、L/R 分離 → VAD → STT に流す。"""
        count = 0
        while not self.capture.queue.empty():
            try:
                stereo = self.capture.queue.get_nowait()
            except Exception:
                break

            left, right = split_channels(stereo)
            if self._swap_lr:
                left, right = right, left

            for chunk in self.vad_l.feed(left):
                self._submit_stt(chunk)
                count += 1

            for chunk in self.vad_r.feed(right):
                self._submit_stt(chunk)
                count += 1

        return count

    def _submit_stt(self, chunk: AudioChunk) -> None:
        """STT をスレッドプールに投入する。"""
        future = self._executor.submit(self.whisper.transcribe, chunk)
        future.add_done_callback(self._on_transcribed)

    def _on_transcribed(self, future) -> None:
        """STT 完了コールバック。"""
        try:
            segment = future.result()
        except Exception:
            logger.exception("STT エラー")
            return

        if segment is None:
            return

        with self._buffer_lock:
            self._transcript_buffer.append(segment)

        # JSONL に永続化
        self.writer.append_transcript(json.dumps(segment.to_dict(), ensure_ascii=False))

        speaker_label = "👤 Interviewer" if segment.speaker == "interviewer" else "🎤 Guest"
        logger.info("%s [%.1fs]: %s", speaker_label, segment.start_time, segment.text[:80])

    def _run_summary_cycle(self) -> None:
        """5分サイクルの要約処理を実行する。"""
        with self._buffer_lock:
            buffer_copy = list(self._transcript_buffer)
            self._transcript_buffer.clear()

        recent_text = self._format_transcript(buffer_copy)

        try:
            response = self.ollama.summarize(
                system_prompt=self._system_prompt,
                hearing_items=self._hearing_items,
                overall_summary_so_far=self._overall_summary,
                recent_transcript=recent_text,
            )
        except Exception:
            logger.exception("Ollama 要約エラー — バッファを次サイクルに繰り越し")
            with self._buffer_lock:
                self._transcript_buffer = buffer_copy + self._transcript_buffer
            return

        # 状態更新
        if response.overall_summary:
            self._overall_summary = response.overall_summary

        # ファイル更新
        self.writer.overwrite_summary(response)

        time_range = ""
        if buffer_copy:
            start = self._format_seconds(buffer_copy[0].start_time)
            end = self._format_seconds(buffer_copy[-1].end_time)
            time_range = f"{start} - {end}"

        self.writer.append_log(
            response.chunk_summary,
            start_time=time_range.split(" - ")[0] if time_range else "",
            end_time=time_range.split(" - ")[1] if " - " in time_range else "",
        )

        logger.info("=== 要約サイクル完了 ===")

    def _format_transcript(self, segments: list[TranscriptSegment]) -> str:
        """セグメントリストを時系列テキストに整形する。"""
        lines = []
        for seg in sorted(segments, key=lambda s: s.start_time):
            speaker = "Interviewer" if seg.speaker == "interviewer" else "Guest"
            time_str = self._format_seconds(seg.start_time)
            lines.append(f"[{time_str}] {speaker}: {seg.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"

    def _extract_purpose(self) -> str:
        """ヒアリング項目の先頭から目的を推定する。"""
        for line in self._hearing_items.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
        return ""

    def _shutdown(self) -> None:
        """安全にリソースを解放する。"""
        logger.info("シャットダウン処理中…")
        self.capture.stop()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.ollama.close()

        # 残りバッファがあれば最終要約
        with self._buffer_lock:
            remaining = list(self._transcript_buffer)
            self._transcript_buffer.clear()

        if remaining:
            logger.info("残りバッファ %d セグメントの最終要約を実行", len(remaining))
            try:
                recent_text = self._format_transcript(remaining)
                response = self.ollama.summarize(
                    system_prompt=self._system_prompt,
                    hearing_items=self._hearing_items,
                    overall_summary_so_far=self._overall_summary,
                    recent_transcript=recent_text,
                )
                self.writer.overwrite_summary(response)
                self.writer.append_log(response.chunk_summary)
            except Exception:
                logger.exception("最終要約に失敗")

        logger.info("シャットダウン完了")
