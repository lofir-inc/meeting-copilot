"""会議後の作り直しスクリプトのテスト（音声とモデルは使わない）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from finalize_meeting import (  # noqa: E402
    apply_corrections,
    gap_audio,
    pick_up_gaps_locally,
    read_jsonl,
    relabel_live_rows,
    sent_live,
    transcribe_self,
)

from src.audio.turn_splitter import TurnSplitConfig  # noqa: E402
from src.audio.vad import VadConfig  # noqa: E402
from src.stt.whisper_client import TranscriptSegment  # noqa: E402

SR = 16000


class FakeWhisper:
    """チャンクの長さをそのまま書き起こす代役。"""

    def transcribe(self, chunk):
        return TranscriptSegment(chunk.speaker, f"{chunk.end_time - chunk.start_time:.1f}秒の発言",
                                 chunk.start_time, chunk.end_time, "")


def test_corrections_are_applied_to_the_nearest_row():
    rows = [{"speaker": "不明話者1", "start_time": 10.0}, {"speaker": "不明話者1", "start_time": 30.0}]
    applied = apply_corrections(rows, [{"start_time": 10.2, "new": "参加者A"}])
    assert applied == 1
    assert rows[0]["speaker"] == "参加者A"
    assert rows[1]["speaker"] == "不明話者1"


def test_corrections_far_from_any_row_are_skipped():
    rows = [{"speaker": "不明話者1", "start_time": 10.0}]
    assert apply_corrections(rows, [{"start_time": 300.0, "new": "参加者A"}]) == 0
    assert rows[0]["speaker"] == "不明話者1"


def test_self_channel_is_transcribed_without_diarization():
    """自分側はマイクを分けているので、話者判定にかけず固定ラベルを付ける。"""
    audio = np.zeros(SR * 4, dtype=np.float32)
    audio[SR : SR * 2] = 0.3   # 1 秒だけ話す
    rows = transcribe_self(audio, SR, VadConfig(silence_duration_sec=0.3, min_chunk_sec=0.2, max_chunk_sec=10), FakeWhisper(), "自分", 0.1)
    assert rows and all(row["speaker"] == "自分" for row in rows)
    assert rows[0]["start_time"] < 2.0


# ------------------------------------------ 方式④（会議中に外で起こしたぶんを使う）

class FakeDiarizer:
    """振幅で話者を決める代役（0.5 → 田中 / それ以外 → 不明話者1）。"""

    def __init__(self):
        self.seen: list[float] = []

    def identify(self, audio, sample_rate, at=0.0):
        from src.audio.enrolled_diarizer import IdentifyResult

        self.seen.append(at)
        peak = float(np.max(np.abs(audio))) if np.asarray(audio).size else 0.0
        if abs(peak - 0.5) < 0.01:
            return IdentifyResult(name="田中", similarity=0.9)
        return IdentifyResult(name="不明話者1", similarity=0.2, is_new=True, confident=False)

    def report_text(self, name, chars):
        pass

    def merge_unknowns(self):
        return []


def test_会議中に外へ送った会議だけ送り直さない(tmp_path):
    """引数なしで呼ばれたときの自動判定。送信の記録がその会議の作り方を覚えている。"""
    assert sent_live(tmp_path) is False
    (tmp_path / "external_sends.jsonl").write_text(
        '{"note": "finalize_meeting"}\n{"note": "live_batch"}\n', encoding="utf-8")
    assert sent_live(tmp_path) is True


def test_壊れた行があっても読める(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nこわれた行\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_話者は当て直すが自分側は触らない(tmp_path):
    """自分側はマイクが別なので声紋にかけない（かけると誤ラベルの元）。"""
    audio = np.full(SR * 10, 0.5, dtype=np.float32)
    rows = [{"speaker": "自分", "text": "そこは", "start_time": 1.0, "end_time": 2.0},
            {"speaker": "不明話者1", "text": "お願いします", "start_time": 3.0, "end_time": 4.0}]

    counts = relabel_live_rows(rows, audio, SR, FakeDiarizer(), "自分")

    assert rows[0]["speaker"] == "自分"      # 触らない
    assert rows[1]["speaker"] == "田中"      # 会議のあとの声紋で当て直す
    assert counts == {"田中": 1}


def test_取りこぼした区間は録音から切り出す():
    audios = {"remote": np.arange(SR * 10, dtype=np.float32), "self": np.zeros(SR * 10, dtype=np.float32)}
    piece = gap_audio({"stream": "remote", "start_time": 2.0, "end_time": 3.0}, audios, SR)
    assert piece.size == SR and piece[0] == pytest.approx(2.0 * SR)
    assert gap_audio({"stream": "remote", "start_time": 50.0, "end_time": 51.0}, audios, SR) is None
    assert gap_audio({"stream": "よその系統", "start_time": 0.0, "end_time": 1.0}, audios, SR) is None


def test_外へ出せなかった区間は手元で拾う():
    """外が駄目でも議事録は埋まる（会議が失われてはいけない）。時刻は会議の時計に戻す。"""
    audio = np.zeros(SR * 120, dtype=np.float32)     # 録音は 2 分ぶん
    audio[SR * 101 : SR * 102] = 0.3                 # 101〜102 秒に 1 秒だけ話す
    gaps = [{"stream": "self", "start_time": 100.0, "end_time": 110.0}]
    vad = VadConfig(silence_duration_sec=0.3, min_chunk_sec=0.2, max_chunk_sec=10)

    rows = pick_up_gaps_locally(gaps, SR, {"self": audio}, vad, FakeDiarizer(), FakeWhisper(),
                                "自分", 0.1, TurnSplitConfig())

    assert rows and all(row["speaker"] == "自分" for row in rows)
    assert 100.0 < rows[0]["start_time"] < 103.0
