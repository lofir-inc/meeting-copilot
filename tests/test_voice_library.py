"""声の台帳 — 前の会議の人を「候補として」出す。"""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from src.audio.voice_library import (
    MATCHES_FILE,
    VoiceLibrary,
    VoiceLibraryConfig,
    read_wav_mono,
    record_match,
    voices_from_session,
)


def _unit(*values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _around(base: np.ndarray, count: int, noise: float = 0.05, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [base + rng.normal(0, noise, size=base.shape).astype(np.float32) for _ in range(count)]


PERSON_C = _unit(1.0, 0.0, 0.0, 0.0)
PERSON_D = _unit(0.0, 1.0, 0.0, 0.0)


@pytest.fixture
def library(tmp_path):
    return VoiceLibrary(tmp_path / "voices" / "library.json", VoiceLibraryConfig(enabled=True))


class TestLearn:
    def test_名前の付いた人を会議ごとに覚えて保存する(self, library, tmp_path):
        result = library.learn("2026-09-14_1359", {"参加者C": _around(PERSON_C, 8)})
        library.save()

        assert result.learned == {"参加者C": 8}
        reloaded = VoiceLibrary(tmp_path / "voices" / "library.json")
        assert reloaded.people["参加者C"][0]["session"] == "2026-09-14_1359"
        assert (tmp_path / "voices" / "library.json").stat().st_mode & 0o777 == 0o600

    def test_話した量が少ない人は覚えない(self, library):
        result = library.learn("s", {"参加者C": _around(PERSON_C, 3)})

        assert not result.learned and "3 件" in result.skipped["参加者C"]

    def test_声がそろわない名前は複数人の混ざりとみなして覚えない(self, library):
        """1 つの名前に 2 人が入っていると、台帳が次の会議で別人を呼び寄せる。"""
        mixed = _around(PERSON_C, 5) + _around(PERSON_D, 5)

        result = library.learn("s", {"参加者C": mixed})

        assert not result.learned and "混ざっている" in result.skipped["参加者C"]

    def test_同じ会議を覚え直したら置き換える(self, library):
        library.learn("s1", {"参加者C": _around(PERSON_C, 6)})
        library.learn("s1", {"参加者C": _around(PERSON_C, 7, seed=1)})

        assert [entry["rows"] for entry in library.people["参加者C"]] == [7]

    def test_会議ごとに持ち上限で古いものから落とす(self, tmp_path):
        library = VoiceLibrary(tmp_path / "l.json", VoiceLibraryConfig(max_meetings_per_person=2))
        for index in range(3):
            library.learn(f"s{index}", {"参加者C": _around(PERSON_C, 6, seed=index)})

        assert [entry["session"] for entry in library.people["参加者C"]] == ["s1", "s2"]


class TestSuggest:
    def test_近い人を近い順に出す(self, library):
        library.learn("s1", {"参加者C": _around(PERSON_C, 6), "参加者D": _around(PERSON_D, 6)})

        found = library.suggest(_unit(0.95, 0.3, 0.0, 0.0))

        assert [item.name for item in found] == ["参加者C"]
        assert found[0].last_session == "s1"

    def test_床より遠い人は出さない(self, library):
        library.learn("s1", {"参加者C": _around(PERSON_C, 6)})

        assert library.suggest(_unit(0.5, 0.5, 0.5, 0.5)) == []

    def test_会議の中にもういる人は出さない(self, library):
        library.learn("s1", {"参加者C": _around(PERSON_C, 6)})

        assert library.suggest(PERSON_C, exclude={"参加者C"}) == []

    def test_いちばん近い会議の声で比べる(self, library):
        """人の声を 1 つに平均しない（会議ごとにマイクや回線が違う）。"""
        library.learn("zoom", {"参加者C": _around(PERSON_C, 6)})
        library.learn("meet", {"参加者C": _around(_unit(0.0, 0.0, 1.0, 0.0), 6)})

        found = library.suggest(_unit(0.0, 0.05, 1.0, 0.0))

        assert found and found[0].name == "参加者C" and found[0].last_session == "meet"


class TestForget:
    def test_消すときはゴミ箱に退避してから外す(self, library, tmp_path):
        library.learn("s1", {"参加者C": _around(PERSON_C, 6)})

        moved = library.forget("参加者C", tmp_path / "trash")

        assert "参加者C" not in library.people
        assert json.loads(moved.read_text(encoding="utf-8"))["name"] == "参加者C"


def test_会議の全文と録音から人ごとの声を集める(tmp_path):
    rows = [
        {"speaker": "自分", "start_time": 0.0, "end_time": 5.0},        # 自分は覚えない
        {"speaker": "参加者C", "start_time": 5.0, "end_time": 9.0},
        {"speaker": "参加者C", "start_time": 9.0, "end_time": 10.0},       # 短い行は使わない
        {"speaker": "不明話者1", "start_time": 10.0, "end_time": 15.0},  # 名前の無い人は覚えない
    ]
    clips = []

    def embed(clip, sample_rate):
        clips.append(len(clip) / sample_rate)
        return np.ones(4, dtype=np.float32)

    voices = voices_from_session(rows, np.zeros(16000 * 20, dtype=np.float32), 16000,
                                 self_name="自分", config=VoiceLibraryConfig(), embed=embed)

    assert list(voices) == ["参加者C"] and clips == [4.0]


def test_録音を読む(tmp_path):
    path = tmp_path / "recording_remote.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes((np.full(480, 16384, dtype=np.int16)).tobytes())

    audio, sample_rate = read_wav_mono(path)

    assert sample_rate == 48000 and audio.shape == (480,) and audio[0] == pytest.approx(0.5)


def test_押した断ったを残す(tmp_path):
    record_match(tmp_path, speaker="不明話者1", suggestion={"name": "参加者C", "score": 0.91}, accepted=False)

    entry = json.loads((tmp_path / MATCHES_FILE).read_text(encoding="utf-8"))
    assert entry["accepted"] is False and entry["score"] == 0.91
