"""mlx-whisper を使用した音声認識クライアント。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.audio.vad import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class SttConfig:
    """音声認識の設定。"""

    engine: str = "local"
    """文字起こしをどこで回すか。`local`（mlx-whisper）| `gemini`（会議中の 30 秒刻みバッチ）。

    `gemini` でも**このクライアントは残る**。承認が取れない・オフライン・課金を確認できない
    ときは手元へ落ちるため（会議が失われてはいけない）。設定は `meeting.external_stt`。
    """

    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "ja"
    beam_size: int = 5
    initial_prompt: str | None = None
    min_avg_logprob: float | None = None
    """これより自信の低い出力を捨てる（Whisper の avg_logprob の最小値で判定）。None なら判定しない。
    会議モードだけが設定する（対面モードの挙動は変えない）。"""


@dataclass
class TranscriptSegment:
    """文字起こし結果の1セグメント。"""
    speaker: str        # "interviewer" or "guest"
    text: str
    start_time: float   # インタビュー開始からの経過秒
    end_time: float
    timestamp: str      # ISO 8601

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timestamp": self.timestamp,
        }


class WhisperClient:
    """mlx-whisper による音声→テキスト変換。"""

    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self._model_loaded = False

    def _ensure_model(self) -> None:
        """初回呼び出し時にモデルをロード（import も遅延）。"""
        if not self._model_loaded:
            import mlx_whisper  # noqa: F401
            logger.info("mlx-whisper model loading: %s", self.config.model)
            # warm-up: 短い無音で初回ロードを済ませる
            mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32),
                path_or_hf_repo=self.config.model,
                language=self.config.language,
            )
            self._model_loaded = True
            logger.info("mlx-whisper model loaded")

    def transcribe(self, chunk: AudioChunk) -> TranscriptSegment | None:
        """AudioChunk をテキスト化して TranscriptSegment を返す。

        テキストが空の場合は None を返す。
        """
        self._ensure_model()
        import mlx_whisper

        # mlx-whisper は 16kHz を期待する場合がある
        audio = chunk.audio
        if chunk.sample_rate != 16000:
            audio = self._resample(audio, chunk.sample_rate, 16000)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.config.model,
            language=self.config.language,
            initial_prompt=self.config.initial_prompt,
            condition_on_previous_text=False,
        )

        text = result.get("text", "").strip()
        if not text:
            return None

        segments = result.get("segments") or []
        if self.config.min_avg_logprob is not None and segments:
            confidence = min(float(seg.get("avg_logprob", 0.0)) for seg in segments)
            if confidence < self.config.min_avg_logprob:
                logger.debug("[%s] low confidence filtered (%.2f): %s", chunk.speaker, confidence, text)
                return None

        # Whisper hallucination フィルタ — 無音・微小音声で出る典型的なゴミを除去
        if self._is_hallucination(text):
            logger.debug("[%s] hallucination filtered: %s", chunk.speaker, text)
            return None
        if self._echoes_prompt(text):
            logger.debug("[%s] initial_prompt echo filtered: %s", chunk.speaker, text)
            return None

        logger.debug("[%s] %.1fs-%.1fs: %s", chunk.speaker, chunk.start_time, chunk.end_time, text)

        return TranscriptSegment(
            speaker=chunk.speaker,
            text=text,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            timestamp=datetime.now().isoformat(),
        )

    # Whisper が無音・微小音に対して出力する典型的な hallucination パターン
    _HALLUCINATION_EXACT = {
        "ご視聴ありがとうございました",
        "次回予告",
        "音楽",
        "おやすみなさい",
        "ありがとうございました",
        "チャンネル登録",
        "お願いします",
        "字幕",
        "MBSニュース",
        "END",
        "Thank you.",
        "Thanks for watching!",
        "Subtitles by",
        "...",
    }

    # 部分一致で除去するパターン
    _HALLUCINATION_CONTAINS = [
        "ご視聴",
        "次回の動画",
        "チャンネル登録",
        "高評価",
        "お気に入り",
        "Subtitles",
        "Subscribe",
        "次回予告",
    ]

    @classmethod
    def _is_hallucination(cls, text: str) -> bool:
        """Whisper の典型的な hallucination かどうか判定する。"""
        cleaned = text.strip().rstrip("。、！!.").strip()
        # 完全一致
        if cleaned in cls._HALLUCINATION_EXACT:
            return True
        # 部分一致
        for pattern in cls._HALLUCINATION_CONTAINS:
            if pattern in cleaned:
                return True
        # 短すぎるテキスト（2文字以下）
        if len(cleaned) <= 2:
            return True
        # 繰り返し検知 — 短い単語・フレーズが3回以上連続するパターン
        import re
        if re.search(r"(.{1,10})\1{2,}", cleaned):
            return True
        # スペース区切りの同一語の繰り返し（「政策 政策 政策」等）
        words = cleaned.split()
        if len(words) >= 3 and len(set(words)) <= 2:
            return True
        return False

    def _echoes_prompt(self, text: str) -> bool:
        """出力が initial_prompt の語（登録名・用語）だけでできているか。

        息や物音の短い区間で、Whisper はヒントをそのまま読み上げる。2026-09-11 の本番では
        全 1,644 行のうち 494 行が「参加者A、参加者B」のような名前だけの行だった。
        語だけを単独で言った本物の発話（「法人番号」だけ、など）も落ちるが、それは稀なので割り切る。
        """
        import re

        prompt = self.config.initial_prompt
        if not prompt:
            return False
        terms = sorted({term.strip() for term in re.split(r"[、,\s]+", prompt) if term.strip()}, key=len, reverse=True)
        rest = text
        for term in terms:
            rest = rest.replace(term, "")
        return rest != text and not re.sub(r"[\s、。,.!?！？・]+", "", rest)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """簡易リサンプリング（線形補間）。"""
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
