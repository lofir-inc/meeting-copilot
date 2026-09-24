"""Deepgram へ音声を送って文字起こしする（会議中の 30 秒刻み／会議のあとの作り直し）。

2026-09-18 に実測して入れた（`PLAN-cloud-hybrid.md` の 8b）。日本語の精度は Gemini・Whisper と
**ほぼ同じ**で、差は速さと費用だった:

    朗読（Common Voice 120 本）   Deepgram 13.3% / Gemini 13.3% / Whisper 13.0%
    会話（YODAS 120 発話）        Deepgram 38.2% / Gemini 39.2% / Whisper 43.0%
    速さ                          Deepgram 252 倍速 / Whisper 50 / Gemini 27
    費用                          Deepgram ¥39/時間 / Gemini ¥83/時間

`TranscribeApi`（Gemini）と**同じ形**にしてある（`transcribe_audio` / `transcribe_file`）。
  差し替えるだけで会議中の経路がそのまま動く。

関門は Gemini と同じ数だけ通る（会議ごとの承認 → **学習に使われない口かの確認** → 記録）。
  Gemini では「課金が有効か」を見るが、Deepgram では **MIP（モデル改善プログラム）から
  抜けているか**を見る（`data_use()`）。禁じている理由（送った内容が製品改善に使われ、
  消せない）に、より直接効くのはこちら。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LISTEN = "https://api.deepgram.com/v1/listen"
PROJECTS = "https://api.deepgram.com/v1/projects"


@dataclass
class DeepgramConfig:
    """`settings.yaml` の `meeting.external_stt`（Deepgram のとき）。"""

    model: str = "nova-3"
    language: str = "ja"
    timeout_sec: int = 300
    smart_format: bool = True
    """句読点と数字の書き方を整える（会議の読みやすさに効く）。"""

    diarize: bool = False
    """話者分け。既定は off — 話者は**手元の声紋**で当てる（実測 97% 対 76%）。"""

    mip_opt_out: bool = True
    """**必ず true。** これを付けた要求は「モデル改善に使わない」だけでなく、
    **保存もされない**（zero data retention。応答を返したら音声も文字起こしも消える）。

    2026-09-18 に判明: これは**リクエストごとのパラメータ**で、プロジェクト設定ではない。
    プロジェクトの `mip_opt_out` を PATCH すると **200 が返るのに値は変わらない**（黙って無視される）。
    「設定したか」ではなく「毎回付けているか」で守る。
    """


class DeepgramApi:
    """Deepgram の口。1 回の呼び出しで 1 つの音声を起こす（Gemini の `TranscribeApi` と同じ形）。"""

    def __init__(self, key: str, config: DeepgramConfig | None = None) -> None:
        if not key:
            raise ValueError("API キーがありません")
        self._key = key            # 表示・ログ禁止
        self.config = config or DeepgramConfig()

    # ------------------------------------------------------------- 関門

    def data_use(self) -> dict:
        """**送ったものが学習に使われず、保存もされない**ことを機械で確かめる（関門）。

        2026-09-18 に Deepgram の仕様を読んで作り直した:

        - opt out は**リクエストごとのパラメータ** `mip_opt_out=true`。付けた要求は
          モデル改善に使われず、**応答を返したあと保存もされない**（zero data retention）
        - プロジェクト設定の `mip_opt_out` を変えようとすると **200 が返るのに変わらない**
          （Pay As You Go では黙って無視される）。だから「設定したか」では守れない

        ここで見るのは 2 つ:

        1. この口が**毎回 opt out を付ける設定**になっているか
        2. その形の要求を Deepgram が**実際に受け付ける**か（0.3 秒の無音で 1 回試す）

        確かめられないときは False（fail-closed）。
        """
        if not self.config.mip_opt_out:
            return {"ok": False, "why": "opt out を付けない設定です（送ったものが学習に使われます）"}
        try:
            import numpy as np

            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as work:
                probe = Path(work) / "probe.wav"
                _write_wav(probe, np.zeros(int(16000 * 0.3), dtype=np.float32), 16000)
                payload = self._post(probe.read_bytes())
        except Exception as error:                    # noqa: BLE001
            return {"ok": False, "why": f"opt out つきの要求を試せませんでした（{error}）"}
        request_id = str((payload.get("metadata") or {}).get("request_id", ""))
        return {"ok": True, "request_id": request_id,
                "why": "毎回 mip_opt_out=true を付けます（学習に使われず、保存もされません）。"
                       f"確認: Deepgram の Usage > Logs で request {request_id[:8]}… が opted out と出ます"}

    # ------------------------------------------------------- 文字起こし

    def transcribe_file(self, path: Path) -> dict:
        """音声 1 つを投げて `{"utterances": [...], "words": [...], "usage": {...}}` を返す。"""
        return self._parse(self._post(Path(path).read_bytes()))

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int, work_dir: Path) -> dict:
        """メモリ上の音声をそのまま起こす（会議中の 30 秒刻みで使う）。

        wav にしてから送る（Deepgram は生の PCM も受けるが、レート違いの事故を避ける）。
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        piece = work_dir / f"window-{time.time_ns()}.wav"
        try:
            _write_wav(piece, audio, sample_rate)
            return self.transcribe_file(piece)
        finally:
            piece.unlink(missing_ok=True)

    # ------------------------------------------------------------- 中身

    def _get(self, url: str) -> dict:
        request = urllib.request.Request(url, headers={"Authorization": f"Token {self._key}",
                                                       "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, body: bytes) -> dict:
        query = urllib.parse.urlencode({
            "model": self.config.model, "language": self.config.language,
            "smart_format": str(self.config.smart_format).lower(),
            "diarize": str(self.config.diarize).lower(),
            "punctuate": "true",
            # 毎回付ける（付けた要求は保存されない）。付け忘れは事故そのものなので設定で切れない
            "mip_opt_out": str(bool(self.config.mip_opt_out)).lower(),
        })
        request = urllib.request.Request(f"{LISTEN}?{query}", data=body, method="POST",
                                         headers={"Authorization": f"Token {self._key}",
                                                  "Content-Type": "audio/wav"})
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"Deepgram が {error.code} を返しました: {detail}") from error

    @staticmethod
    def _parse(payload: dict) -> dict:
        """語と発話を、会議中の経路が読める形にそろえる。"""
        channels = (payload.get("results") or {}).get("channels") or []
        alternatives = (channels[0].get("alternatives") if channels else []) or []
        best = alternatives[0] if alternatives else {}
        words = [{"text": str(one.get("punctuated_word") or one.get("word") or ""),
                  "start": float(one.get("start", 0) or 0), "end": float(one.get("end", 0) or 0),
                  "speaker": str(one.get("speaker", "")) if one.get("speaker") is not None else ""}
                 for one in (best.get("words") or [])]
        text = str(best.get("transcript", "")).strip()
        utterances = [{"text": text, "start": words[0]["start"] if words else 0.0,
                       "end": words[-1]["end"] if words else 0.0, "speaker": ""}] if text else []
        duration = float((payload.get("metadata") or {}).get("duration", 0) or 0)
        return {"utterances": utterances, "words": words,
                "usage": {"audioSeconds": duration}}


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """float32 の音声を 16bit の wav にする。"""
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes((samples * 32767).astype(np.int16).tobytes())
