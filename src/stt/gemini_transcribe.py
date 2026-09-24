"""Gemini の文字起こし専用モデルを呼ぶところ（HTTP の口だけ）。

口の形は 2026-09-13 に実測で確定した。詳しい経緯は `scripts/gemini_transcribe.py` の冒頭。
  要点だけ再掲する:

  - 専用モデル（`gemini-3.5-transcribe`）は **`generateContent` では何も返さない**。
    正しい入口は `POST /v1beta/interactions`、音声は **Files API に上げて uri で渡す**
    （インライン base64 は不可）。返るのは語ごとの `word_info`（話者・開始・終了）
  - 話者分け／語の時刻を付けると音声は 30 分まで（付けなければ 1 時間）
  - 用語リストと時刻は併用できない（`custom_vocabulary is incompatible with timestamps.` の 400）

ここには**関門を置かない**。課金の確認と会議ごとの承認は呼び出し側
（`src/stt/external_consent.py`）の仕事で、ここへ来た時点で通っている前提。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BASE = "https://generativelanguage.googleapis.com"
INTERACTIONS = BASE + "/v1beta/interactions"


@dataclass
class TranscribeConfig:
    """専用モデルへの投げ方。"""

    model: str = "gemini-3.5-transcribe"
    language: str = "ja-JP"
    timeout_sec: float = 900.0
    diarize: bool = False
    """話者分けも外に任せるか。既定は false（手元の声紋のほうが 96% 対 76% で強い）。"""
    timestamps: bool = True
    """語ごとの時刻をもらうか。false のときだけ `vocabulary` が効く。"""
    vocabulary: list[str] = field(default_factory=list)
    max_gap_sec: float = 0.8
    """語を発話にまとめるときの無音の切れ目（こちらの VAD と同じ考え方）。"""


def load_key(key_file: str | Path) -> str:
    """キーの 1 行目だけを読む。中身は返すだけで、ログにも画面にも出さない。"""
    path = Path(key_file).expanduser()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    key = lines[0].strip() if lines else ""
    if not key:
        raise ValueError(f"{path} が空です")
    return key


def encode_mp3(audio: np.ndarray, sample_rate: int, out_path: Path, *, bitrate: str = "64k") -> Path:
    """float32 のモノラル音声を mp3 にする。トークンは長さで決まるので音質は上げない。"""
    raw = np.asarray(audio, dtype=np.float32).reshape(-1).tobytes()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
         "-ac", "1", "-b:a", bitrate, str(out_path)],
        input=raw, check=True,
    )
    return out_path


def retry_after_seconds(detail: str, attempt: int) -> float:
    """429 の本文にある retryDelay（例 "31s"）を読む。無ければ指数的に待つ。"""
    found = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', detail)
    return float(found.group(1)) + 1.0 if found else min(60.0 * 2 ** attempt, 600.0)


def post(url: str, data: bytes, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def with_retry(call, *, retries: int = 4, sleep=time.sleep, on_wait=None):
    """429 だけは待って繰り返す（1 分あたりの上限に当たったとき）。他は例外にする。"""
    for attempt in range(retries + 1):
        code, headers, body = call()
        if code < 400:
            return headers, body
        if code != 429 or attempt == retries:
            raise RuntimeError(f"HTTP {code}: {body.decode(errors='replace')[:300]}")
        wait = retry_after_seconds(body.decode(errors="replace"), attempt)
        if on_wait is not None:
            on_wait(wait)
        logger.warning("Gemini の上限に当たりました。%.0f 秒待って再試行します", wait)
        sleep(wait)
    raise AssertionError("到達しない")


def offset_seconds(value: str) -> float:
    """"0.100s" / "8s" を秒に直す。"""
    return float(str(value).rstrip("s") or 0.0)


def words_of(annotations: list[dict]) -> list[dict]:
    """語ごとの注記を、まとめずにそのまま取り出す。

    会議中の 30 秒刻み（`src/stt/live_batch.py`）は**無音を落として繋いで**送るので、
    送った音声の中では話者交代の間が消えている。まとめるのは、繋ぎ目を知っている側の仕事。
    """
    return [{"speaker": str(item.get("speaker") or ""),
             "text": str(item.get("text", "")),
             "start": offset_seconds(item.get("start_offset", "0s")),
             "end": offset_seconds(item.get("end_offset", "0s"))}
            for item in annotations if item.get("type") == "word_info"]


def words_to_utterances(annotations: list[dict], *, max_gap: float = 0.8) -> list[dict]:
    """語ごとの word_info を発話の単位にまとめる。

    話者が替わったとき、または無音が `max_gap` 秒を超えたところで切る
    （こちらの VAD の silence_duration_sec と同じ考え方にしておく）。
    """
    rows: list[dict] = []
    for item in annotations:
        if item.get("type") != "word_info":
            continue
        speaker = str(item.get("speaker") or "")
        start = offset_seconds(item.get("start_offset", "0s"))
        end = offset_seconds(item.get("end_offset", "0s"))
        text = str(item.get("text", ""))
        if rows and rows[-1]["speaker"] == speaker and start - rows[-1]["end"] <= max_gap:
            rows[-1]["text"] += text
            rows[-1]["end"] = end
        else:
            rows.append({"speaker": speaker, "text": text, "start": start, "end": end})
    return [row for row in rows if row["text"].strip()]


class TranscribeApi:
    """文字起こし専用モデルの口。1 回の呼び出しで 1 つの音声ファイルを起こす。"""

    def __init__(self, key: str, config: TranscribeConfig | None = None) -> None:
        if not key:
            raise ValueError("API キーがありません")
        self._key = key            # 表示・ログ禁止
        self.config = config or TranscribeConfig()

    # -------------------------------------------------------------- Files API

    def upload(self, path: Path, mime: str = "audio/mp3") -> str:
        """Files API に音声を上げて uri を返す。専用モデルはインライン base64 を受けない。"""
        blob = Path(path).read_bytes()
        headers, _ = with_retry(lambda: post(
            f"{BASE}/upload/v1beta/files",
            json.dumps({"file": {"display_name": Path(path).name}}).encode(),
            {"x-goog-api-key": self._key, "Content-Type": "application/json",
             "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
             "X-Goog-Upload-Header-Content-Length": str(len(blob)),
             "X-Goog-Upload-Header-Content-Type": mime},
            self.config.timeout_sec))
        upload_url = headers.get("X-Goog-Upload-URL") or headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("アップロード先の URL が返りませんでした")
        _, body = with_retry(lambda: post(
            upload_url, blob,
            {"Content-Length": str(len(blob)), "X-Goog-Upload-Offset": "0",
             "X-Goog-Upload-Command": "upload, finalize"},
            self.config.timeout_sec))
        return json.loads(body)["file"]["uri"]

    def delete(self, uri: str, timeout: float = 60.0) -> None:
        """上げた音声を消す（48 時間で自動的に消えるが、置いたままにしない）。"""
        name = uri.rsplit("/files/", 1)[-1]
        request = urllib.request.Request(f"{BASE}/v1beta/files/{name}", method="DELETE",
                                         headers={"x-goog-api-key": self._key})
        try:
            urllib.request.urlopen(request, timeout=timeout).close()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            pass    # 消せなくても本題ではない（期限で消える）

    # ------------------------------------------------------------- 文字起こし

    def transcribe_file(self, path: Path) -> dict:
        """音声 1 つを投げて `{"utterances": [...], "usage": {...}}` を返す。"""
        uri = self.upload(Path(path))
        try:
            _, raw = with_retry(lambda: post(
                INTERACTIONS, json.dumps(self._body(uri)).encode(),
                {"x-goog-api-key": self._key, "Content-Type": "application/json"},
                self.config.timeout_sec))
        finally:
            self.delete(uri)
        return self._parse(json.loads(raw))

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int, work_dir: Path) -> dict:
        """メモリ上の音声をそのまま起こす（会議中の 30 秒刻みで使う）。"""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        piece = work_dir / f"window-{time.time_ns()}.mp3"
        try:
            encode_mp3(audio, sample_rate, piece)
            return self.transcribe_file(piece)
        finally:
            piece.unlink(missing_ok=True)

    # ------------------------------------------------------------------ 内部

    def _body(self, uri: str) -> dict:
        # mode は verbatim（smart は時刻・話者と併用できない）
        mode: dict = {"type": "verbatim"}
        if self.config.timestamps:
            mode["timestamp_granularities"] = ["word"]
        if self.config.diarize:
            mode["diarization_mode"] = "speaker"
        transcription: dict = {"language_codes": [self.config.language], "mode": mode}
        # 用語リストと時刻は同時に指定できない（400）。冒頭の注記参照
        if self.config.vocabulary and not self.config.timestamps:
            transcription["custom_vocabulary"] = list(self.config.vocabulary)
        return {"model": self.config.model,
                "input": [{"type": "audio", "uri": uri, "mime_type": "audio/mp3"}],
                "generation_config": {"transcription_config": transcription}}

    def _parse(self, payload: dict) -> dict:
        annotations: list[dict] = []
        plain = ""
        for step in payload.get("steps", []):
            for content in step.get("content", []):
                annotations += content.get("annotations", []) or []
                plain += content.get("text", "") or ""
        # 時刻を頼まなかったときは語の注記が来ない。本文だけを 1 行として返す
        utterances = (words_to_utterances(annotations, max_gap=self.config.max_gap_sec) if annotations
                      else ([{"speaker": "", "text": plain, "start": 0.0, "end": 0.0}] if plain.strip() else []))
        usage = payload.get("usage", {})
        return {
            "utterances": utterances,
            "words": words_of(annotations),
            "usage": {"promptTokenCount": usage.get("total_input_tokens", 0),
                      "candidatesTokenCount": usage.get("total_output_tokens", 0)},
        }
