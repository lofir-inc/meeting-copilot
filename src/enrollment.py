"""会議冒頭の話者登録フロー（CLI）。

「お一人ずつ、お名前と一言お願いします」を回して、参加者の声紋を名前つきで登録する。
結果は `workspace/sessions/<session>/speakers.json` に保存するので、**再起動しても
登録し直さなくていい**。

登録するのは**相手側（ループバック）だけ**。自分の声はマイクを物理的に分けている
ので判定不要（固定ラベル）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd

from src.audio.devices import LevelReading, level_of
from src.audio.enrolled_diarizer import DiarizerConfig, MIN_DURATION_SEC, EnrolledDiarizer

logger = logging.getLogger(__name__)


class FrameSource(Protocol):
    """登録用のモノラル音声を秒数指定で返す取得元。"""

    def read_seconds(self, seconds: float) -> np.ndarray:
        """指定秒数のモノラル音声を返す。"""


class CaptureFrameSource:
    """MultiCapture の特定ソースを登録用の取得元として包む。"""

    def __init__(self, capture: object, key: str, sample_rate: int) -> None:
        self.capture = capture
        self.key = key
        self.sample_rate = sample_rate

    def read_seconds(self, seconds: float) -> np.ndarray:
        """キャプチャキューから指定秒数の音声を返す。"""
        return self.capture.record_from(self.key, seconds)


@dataclass
class EnrollmentConfig:
    """話者登録の設定。"""

    duration_sec: float = 5.0
    sample_rate: int = 48000
    similarity_threshold: float = 0.75
    max_adapt_embeddings: int = 10
    diarizer: DiarizerConfig = field(default_factory=DiarizerConfig)

    # 録音レベルがこれを下回ったら「声が入っていない」とみなして録り直しを促す
    min_peak_dbfs: float = -60.0

    min_voiced_sec: float = 1.5
    """録音のうち声が入っていた長さの下限。これより短いと録り直しを促す。

    2026-09-11 本番: 参加者B の 5 秒には「あはい ここです」の末尾 0.4 秒しか声が入っておらず、
    物音が 参加者B として登録された。本人は「不明話者1」に回り（MacWhisper と照合して 94%）、
    参加者B の名前は物音・相づち 104 行に付いた。参加者A（2.1 秒）は 92% 正しく付いた。
    """


def record_sample(
    device: int,
    channels: int,
    seconds: float,
    sample_rate: int,
) -> tuple[np.ndarray, LevelReading]:
    """指定デバイスから数秒録音し、モノラル配列と入力レベルを返す。"""
    frames = int(seconds * sample_rate)
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()

    array = np.asarray(audio, dtype=np.float32)
    mono = array[:, 0].copy() if array.ndim == 2 and array.shape[1] == 1 else (
        array.mean(axis=1).astype(np.float32) if array.ndim == 2 else array
    )
    return mono, level_of(mono, seconds)


def load_or_create(
    speakers_path: Path,
    config: EnrollmentConfig,
) -> tuple[EnrolledDiarizer, bool]:
    """既存の `speakers.json` があれば読み込む。返り値は (diarizer, 読み込んだか)。"""
    if speakers_path.exists():
        try:
            diarizer = EnrolledDiarizer.load(
                speakers_path,
                similarity_threshold=config.diarizer.similarity_threshold,
                max_adapt_embeddings=config.diarizer.max_adapt_embeddings,
                adapt=config.diarizer.adapt,
            )
            return diarizer, True
        except Exception:
            logger.exception("speakers.json の読み込みに失敗。新規に作り直します: %s", speakers_path)

    return (
        EnrolledDiarizer(config.diarizer),
        False,
    )


def run_enrollment(
    source: FrameSource,
    speakers_path: Path,
    config: EnrollmentConfig,
) -> EnrolledDiarizer:
    """対話的な話者登録を実行し、`speakers.json` に保存した diarizer を返す。"""
    diarizer, loaded = load_or_create(speakers_path, config)

    if loaded and len(diarizer):
        print()
        print("=" * 60)
        print("【話者登録】既存の登録が見つかりました")
        print(f"  {speakers_path}")
        for name in diarizer.enrolled_names:
            print(f"    ・{name}")
        if diarizer.unknown_names:
            print(f"  （暫定話者: {', '.join(diarizer.unknown_names)}）")
        print("=" * 60)
        print("  [1] この登録のまま開始する")
        print("  [2] 参加者を追加で登録する")
        print("  [3] 破棄して最初から登録し直す")
        choice = _prompt("選択 [1]: ").strip() or "1"

        if choice == "1":
            return diarizer
        if choice == "3":
            diarizer = EnrolledDiarizer(config.diarizer)

    print()
    print("=" * 60)
    print("【話者登録】")
    print("  参加者に「お一人ずつ、お名前と一言」お願いしてください。")
    print(f"  名前を入力 → Enter で {config.duration_sec:.0f} 秒録音します。")
    print("  空 Enter で登録を終了して本編を開始します。")
    print("  登録しなくても始められます。会議中に画面の行の ✎ で名前を付けると、その声を覚えて")
    print("    以後の発話と、それまでの「不明話者?」の行にも名前が付きます。")
    print("=" * 60)

    while True:
        try:
            name = _prompt("\n話者名（空 Enter で終了）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n登録を中断しました。")
            break

        if not name:
            break

        if not _enroll_one(diarizer, name, source, config):
            continue

    if len(diarizer):
        diarizer.save(speakers_path)
        print(f"\n登録完了: {', '.join(diarizer.enrolled_names)}")
        print(f"保存先: {speakers_path}")
    else:
        print("\n⚠ 誰も登録されていません。")
        print("  相手側の発話はすべて「不明話者N」として記録されます（後から名寄せできます）。")

    return diarizer


def _enroll_one(
    diarizer: EnrolledDiarizer,
    name: str,
    source: FrameSource,
    config: EnrollmentConfig,
) -> bool:
    """1人分を録音して登録する。成功したら True。"""
    while True:
        print(f"  {name} さん、どうぞ… ({config.duration_sec:.0f} 秒録音中)")
        audio = source.read_seconds(config.duration_sec)
        level = level_of(audio, config.duration_sec)
        print(f"  録音レベル: {level}")

        if level.peak_dbfs < config.min_peak_dbfs:
            print("  ⚠ ほとんど音が入っていません。")
            print("    会議アプリの出力先が複数出力装置になっているか確認してください。")
            if not _confirm("  録り直しますか [Y/n]: "):
                return False
            continue

        voiced = voiced_seconds(audio, config.sample_rate)
        if voiced < config.min_voiced_sec:
            print(f"  ⚠ 声が入っていたのは {voiced:.1f} 秒だけです（{config.min_voiced_sec:g} 秒以上必要）。")
            print(f"    {name} さんに、Enter のあと続けて {config.duration_sec:.0f} 秒ほど話してもらってください。")
            if not _confirm("  録り直しますか [Y/n]: "):
                return False
            continue

        try:
            diarizer.enroll(name, audio, config.sample_rate)
        except ValueError as exc:
            print(f"  ⚠ 登録できませんでした: {exc}")
            if not _confirm("  録り直しますか [Y/n]: "):
                return False
            continue

        print(f"  ✅ {name} を登録しました（登録済み: {len(diarizer.enrolled_names)} 名）")
        _warn_similar_voices(diarizer, config.diarizer.enroll_warn_similarity)
        return True


def voiced_seconds(audio: np.ndarray, sample_rate: int, block_sec: float = 0.1, floor_dbfs: float = -40.0) -> float:
    """0.1 秒ごとの音量で、声が入っていた長さを数える（-40 dBFS か、最大から 30 dB 下の高い方を超えた区間）。"""
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    block = int(sample_rate * block_sec)
    count = flat.size // block if block else 0
    if count == 0:
        return 0.0
    db = 10.0 * np.log10(np.mean(flat[: count * block].reshape(count, block) ** 2, axis=1) + 1e-12)
    threshold = max(floor_dbfs, float(db.max()) - 30.0)
    return float(np.sum(db > threshold)) * block_sec


def _warn_similar_voices(diarizer: EnrolledDiarizer, threshold: float) -> None:
    """登録済み話者どうしの声紋が近すぎるときだけ警告する。"""
    for (left, right), similarity in diarizer.pairwise_similarity().items():
        if similarity >= threshold:
            print(
                f"  ⚠ {left} と {right} の声が似すぎています（{similarity:.2f}）。"
                "録り直すか、片方は不明話者として扱うことを検討してください"
            )


def _prompt(message: str) -> str:
    return input(message)


def failed_enrollment_lines(diarizer: EnrolledDiarizer, speakers_path: Path | str) -> list[str]:
    """声の登録が効いていないときに出す文面を返す（空なら問題なし）。

    会議の終わりだけでなく、**録音からの作り直しでも要る**。作り直しは古い
    `speakers.json` をそのまま引き継ぐので、失敗した登録が何度でも効いてしまう。
    2026-09-13 に実際に起きた: 09-11 の会議を作り直すと、幻の「参加者B」が
    224 発話（3.6 文字/発話・95% が相づち）を吸い、本人は `不明話者1` として
    別人に見えていた。検知は実装してあったのに、作り直しの経路へ繋いでいなかった。
    """
    failures = diarizer.failed_enrollments()
    if not failures:
        return []
    lines = ["※ 声の登録が効いていない可能性があります。"]
    for failure in failures:
        lines.append(f"  「{failure['name']}」はこの会議でほぼ当たっていません"
                     f"（{failure['utterances']} 発話・{failure['chars_per_utterance']} 文字/発話）。")
    lines.append(f"  代わりに {'、'.join(failures[0]['candidates'])} が中身のある発話を担っています。")
    lines.append("  人違いなら、登録し直すか名寄せしてください:")
    lines.append(f"    {speakers_path}")
    return lines


def _confirm(message: str, default: bool = True) -> bool:
    try:
        answer = input(message).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not answer:
        return default
    return answer in {"y", "yes", "は", "はい"}


__all__ = [
    "EnrollmentConfig",
    "failed_enrollment_lines",
    "CaptureFrameSource",
    "FrameSource",
    "load_or_create",
    "record_sample",
    "run_enrollment",
    "MIN_DURATION_SEC",
]
