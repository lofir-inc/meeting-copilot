"""登録済み N 話者への最近傍照合の単体テスト。

resemblyzer のロードは重いので、`embed()` を差し替えて既知のベクトルを流す。
判定ロジックそのものを検証する。
"""

import numpy as np
import pytest

from src.audio.enrolled_diarizer import (
    DiarizerConfig,
    UNKNOWN_PREFIX,
    EnrolledDiarizer,
    peak_normalize,
    resample,
)

SR = 48000


def _vec(*values: float) -> np.ndarray:
    """単位ベクトル化した embedding もどき。"""
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# 互いに直交する3人分＋Aに酷似した1本
ALICE = _vec(1, 0, 0, 0)
BOB = _vec(0, 1, 0, 0)
CAROL = _vec(0, 0, 1, 0)
DAVE = _vec(0, 0, 0, 1)
ALICE_AGAIN = _vec(0.99, 0.14, 0, 0)  # cos(ALICE) ≈ 0.990


def long(vec: np.ndarray, seconds: float = 5.0) -> np.ndarray:
    """新規話者を起こせる長さまで引き伸ばす。

    fake_embed は先頭4要素しか見ないので、繰り返しても embedding は変わらない。
    実データでは「相槌のような短い発話が新しい話者を起こしてしまう」ことが問題だったので、
    テストでも**長さを持った配列**を渡して現実に合わせる。
    """
    reps = int(seconds * SR / len(vec)) + 1
    return np.tile(vec, reps)[: int(seconds * SR)].astype(np.float32)


def short(vec: np.ndarray, seconds: float = 1.4) -> np.ndarray:
    """相槌くらいの長さ（実データの不明話者の長さ中央値は 1.4 秒だった）。"""
    return long(vec, seconds)


@pytest.fixture
def diarizer(monkeypatch):
    """embed() を「渡された配列の先頭4要素をそのまま embedding として返す」実装に差し替える。"""

    def fake_embed(audio, sample_rate):
        array = np.asarray(audio, dtype=np.float32)
        if array.size < 4:
            return None
        return array[:4] / np.linalg.norm(array[:4])

    monkeypatch.setattr(EnrolledDiarizer, "embed", staticmethod(fake_embed))
    return EnrolledDiarizer(
        DiarizerConfig(
            similarity_threshold=0.75,
            min_unknown_sec=0.0,
            min_unknown_utterances=1,
        )
    )


class TestEnroll:
    def test_enroll_registers_by_name(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        diarizer.enroll("ボブ", BOB, SR)
        assert diarizer.enrolled_names == ["アリス", "ボブ"]
        assert len(diarizer) == 2

    def test_enroll_same_name_twice_accumulates(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        diarizer.enroll("アリス", ALICE_AGAIN, SR)
        assert len(diarizer) == 1
        assert diarizer.enrolled_names == ["アリス"]

    def test_blank_name_rejected(self, diarizer):
        with pytest.raises(ValueError):
            diarizer.enroll("   ", ALICE, SR)

    def test_unusable_audio_rejected(self, diarizer):
        with pytest.raises(ValueError):
            diarizer.enroll("アリス", np.zeros(2, dtype=np.float32), SR)


class TestIdentify:
    def test_identifies_enrolled_speaker(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        diarizer.enroll("ボブ", BOB, SR)

        result = diarizer.identify(long(ALICE_AGAIN), SR)
        assert result.name == "アリス"
        assert result.confident
        assert not result.is_new
        assert result.similarity > 0.9

    def test_third_speaker_is_not_absorbed(self, diarizer):
        """2話者固定版との決定的な違い。

        旧 diarizer.py は「どちらにも似ていない場合、より近い方に分類」するため
        3人目が必ず A か B に吸い込まれた。こちらは新しい人として起こす。
        """
        diarizer.enroll("アリス", ALICE, SR)
        diarizer.enroll("ボブ", BOB, SR)

        result = diarizer.identify(long(CAROL), SR)
        assert result.name not in {"アリス", "ボブ"}
        assert result.name.startswith(UNKNOWN_PREFIX)
        assert result.is_new
        assert not result.confident

    def test_fourth_speaker_gets_a_distinct_label(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        first = diarizer.identify(long(CAROL), SR)
        second = diarizer.identify(long(DAVE), SR)
        assert first.name != second.name
        assert {first.name, second.name} == {f"{UNKNOWN_PREFIX}1", f"{UNKNOWN_PREFIX}2"}

    def test_unknown_speaker_is_recognised_on_return(self, diarizer):
        """暫定話者を握り潰さないので、同じ人が戻ってくれば同じラベルが付く。"""
        diarizer.enroll("アリス", ALICE, SR)

        first = diarizer.identify(long(CAROL), SR)
        second = diarizer.identify(long(CAROL), SR)
        assert first.name == second.name
        assert not second.is_new
        assert len(diarizer.unknown_names) == 1

    def test_all_speakers_are_enrolled_so_none_unknown(self, diarizer):
        for name, vec in [("アリス", ALICE), ("ボブ", BOB), ("キャロル", CAROL), ("デイブ", DAVE)]:
            diarizer.enroll(name, vec, SR)

        for name, vec in [("アリス", ALICE), ("ボブ", BOB), ("キャロル", CAROL), ("デイブ", DAVE)]:
            assert diarizer.identify(long(vec), SR).name == name
        assert diarizer.unknown_names == []

    def test_identify_with_no_enrollment_creates_unknowns(self, diarizer):
        result = diarizer.identify(long(ALICE), SR)
        assert result.name == f"{UNKNOWN_PREFIX}1"
        assert result.is_new

    def test_too_short_audio_does_not_create_a_speaker(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        result = diarizer.identify(np.zeros(2, dtype=np.float32), SR)
        assert not result.confident
        assert not result.is_new
        assert len(diarizer) == 1  # 話者は増えていない

    def test_threshold_is_respected(self, monkeypatch):
        def fake_embed(audio, sample_rate):
            array = np.asarray(audio, dtype=np.float32)
            return array[:4] / np.linalg.norm(array[:4])

        monkeypatch.setattr(EnrolledDiarizer, "embed", staticmethod(fake_embed))

        # cos(ALICE, ALICE_AGAIN) ≈ 0.990 なので 0.999 の閾値は通らない
        # 候補プールは「合計 6 秒・2 発話」で昇格するので、5 秒の発話を 2 回流す
        strict = EnrolledDiarizer(similarity_threshold=0.999)
        strict.enroll("アリス", ALICE, SR)
        assert not strict.identify(long(ALICE_AGAIN), SR, at=0.0).is_new
        assert strict.identify(long(ALICE_AGAIN), SR, at=5.0).is_new

        loose = EnrolledDiarizer(similarity_threshold=0.5)
        loose.enroll("アリス", ALICE, SR)
        assert loose.identify(long(ALICE_AGAIN), SR).name == "アリス"

    def test_adapt_disabled_keeps_centroid_fixed(self, monkeypatch):
        def fake_embed(audio, sample_rate):
            array = np.asarray(audio, dtype=np.float32)
            return array[:4] / np.linalg.norm(array[:4])

        monkeypatch.setattr(EnrolledDiarizer, "embed", staticmethod(fake_embed))
        d = EnrolledDiarizer(similarity_threshold=0.5, adapt=False)
        d.enroll("アリス", ALICE, SR)
        d.identify(long(ALICE_AGAIN), SR)
        # 取り込んでいないので登録時の1本のまま
        assert d._speakers["アリス"].sample_count == 1


class TestShortChunksDoNotSpawnSpeakers:
    """実データで見つかった一番大きい問題への回帰テスト。

    2026-09-03 の1時間・2話者の録画で、不明話者が**12人**生成された。
    正体は相槌・息継ぎ・ノイズで、長さ中央値 1.4 秒・文字起こし結果は中央値で空文字。
    議事録に1文字も寄与しないのに登録簿を汚し、speakers.json に保存されて
    次回まで持ち越されていた。
    """

    def test_one_short_unmatched_chunk_stays_pending(self):
        d = EnrolledDiarizer(DiarizerConfig(min_unknown_sec=6.0, min_unknown_utterances=2))
        result = d.identify_embedding(CAROL, 1.4, 0.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"
        assert not result.is_new
        assert d.unknown_names == []

    def test_two_utterances_totalling_six_seconds_promote(self):
        d = EnrolledDiarizer(DiarizerConfig(min_unknown_sec=6.0, min_unknown_utterances=2))
        d.identify_embedding(CAROL, 3.0, 0.0)
        result = d.identify_embedding(CAROL, 3.0, 3.0)
        assert result.name == f"{UNKNOWN_PREFIX}1"
        assert result.is_new

    def test_unknown_clusters_merge_when_similar(self):
        """同じ人が 2 つのクラスタに割れたら併合する。

        ただし「数発話ぶん待ってから」（2026-09-11: 昇格直後に併合して別人が 1 人になった）。
        ここでは断片どうし（1 発話あたりの文字が少ない）なので 0.75 で併合される。
        """
        d = EnrolledDiarizer(
            DiarizerConfig(
                similarity_threshold=0.9,
                merge_threshold=0.75,
                min_unknown_sec=0.0,
                min_unknown_utterances=1,
                merge_min_utterances=2,
            )
        )
        first = d.identify_embedding(CAROL, 6.0, 0.0)
        second = d.identify_embedding(_vec(0, 0, 0.8, 0.6), 6.0, 1.0)
        assert first.name != second.name
        for name in (first.name, second.name):
            for _ in range(2):
                d.report_text(name, 3)
        assert d.merge_unknowns()
        assert len(d.unknown_names) == 1

    def test_promotion_of_a_non_first_pending_cluster_does_not_crash(self):
        """回帰テスト（aa の実機検査 2026-09-08）。

        候補プールに 2 件以上たまった状態で、先頭でない候補が昇格すると
        `list.remove` が dataclass の __eq__ を呼び、numpy 配列の比較で落ちていた。
        3 人目の昇格＝本番経路でそのまま例外になる。
        """
        d = EnrolledDiarizer(DiarizerConfig(min_unknown_sec=6.0, min_unknown_utterances=2))
        d.identify_embedding(CAROL, 3.0, 0.0)   # 候補 1（先頭）
        d.identify_embedding(DAVE, 3.0, 1.0)    # 候補 2
        result = d.identify_embedding(DAVE, 3.0, 2.0)   # 先頭でない候補 2 が昇格
        assert result.is_new
        assert result.name == f"{UNKNOWN_PREFIX}1"
        assert len(d._pending) == 1             # 候補 1 は残っている

    def test_pending_cluster_expires_after_ttl(self):
        d = EnrolledDiarizer(DiarizerConfig(pending_ttl_sec=10.0, min_unknown_sec=6.0))
        d.identify_embedding(CAROL, 3.0, 0.0)
        result = d.identify_embedding(CAROL, 3.0, 11.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"
        assert d.unknown_names == []

    def test_adapt_margin_and_enroll_weight_protect_centroid(self, monkeypatch):
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer(
            DiarizerConfig(similarity_threshold=0.5, adapt_margin=0.10, enroll_weight=0.5)
        )
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", ALICE_AGAIN, SR)
        d.identify_embedding(ALICE_AGAIN, 3.0, 0.0)
        assert d._speakers["アリス"].adapt_embeddings == []
        d._speakers.pop("ボブ")
        for _ in range(10):
            d.identify_embedding(ALICE_AGAIN, 3.0, 1.0)
        centroid = d._speakers["アリス"].centroid(10, 0.5)
        assert np.dot(centroid, ALICE) / np.linalg.norm(centroid) >= 0.9

    @staticmethod
    def _with_fake_embed(monkeypatch) -> None:
        """enroll() が 4 要素ベクトルをそのまま embedding として受けるようにする。"""
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )

    def test_short_run_does_not_enter_pending_pool(self, monkeypatch):
        self._with_fake_embed(monkeypatch)
        d = EnrolledDiarizer(DiarizerConfig(similarity_threshold=0.9))
        d.enroll("アリス", ALICE, SR)
        result = d.identify_embedding(CAROL, 2.0, 0.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"
        assert d._pending == []

    def test_short_run_assigns_known_speaker_without_confidence(self, monkeypatch):
        self._with_fake_embed(monkeypatch)
        d = EnrolledDiarizer(DiarizerConfig(similarity_threshold=0.999))
        d.enroll("アリス", ALICE, SR)
        result = d.identify_embedding(ALICE_AGAIN, 2.0, 0.0)
        assert result.name == "アリス"
        assert not result.confident

    def test_short_run_below_short_threshold_stays_unknown(self, monkeypatch):
        self._with_fake_embed(monkeypatch)
        d = EnrolledDiarizer(DiarizerConfig(short_turn_threshold=0.65))
        d.enroll("アリス", ALICE, SR)
        result = d.identify_embedding(CAROL, 2.0, 0.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"

    def test_short_run_with_small_margin_stays_unknown(self, monkeypatch):
        """aa の指摘: 別人 2 名が 0.67 で並ぶ空間では、床だけだと他人に付く。差で決める。"""
        self._with_fake_embed(monkeypatch)
        d = EnrolledDiarizer(
            DiarizerConfig(similarity_threshold=0.9, short_turn_threshold=0.65, short_turn_margin=0.10)
        )
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", BOB, SR)
        # アリス 0.78 / ボブ 0.74 → 差 0.04 → 付けない
        result = d.identify_embedding(_vec(0.78, 0.74, 0, 0), 2.0, 0.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"
        # アリス 0.78 / ボブ 0.62 → 差 0.16 → 付ける
        result = d.identify_embedding(_vec(0.78, 0.62, 0, 0), 2.0, 1.0)
        assert result.name == "アリス"
        assert not result.confident

    def test_short_run_does_not_adapt(self, monkeypatch):
        self._with_fake_embed(monkeypatch)
        d = EnrolledDiarizer(DiarizerConfig(similarity_threshold=0.75))
        d.enroll("アリス", ALICE, SR)
        d.identify_embedding(ALICE_AGAIN, 2.0, 0.0)
        assert d._speakers["アリス"].adapt_embeddings == []

    def test_short_run_is_not_attached_to_a_promoted_unknown(self):
        """短い run は登録済み話者にだけ当てる。昇格済みの不明話者には当てない。

        当てると断片が不明クラスタを吸い寄せて太らせる（08-26 の 不明話者7＝188 発話・
        中央 2.0 秒・幻覚まじり。aa 実測 2026-09-08）。本物の 3 人目は 3 秒以上の run で
        昇格するので、この制限で失われない。
        """
        d = EnrolledDiarizer(
            DiarizerConfig(min_unknown_sec=0.0, min_unknown_utterances=1)
        )
        promoted = d.identify_embedding(CAROL, 3.0, 0.0)
        assert promoted.name == f"{UNKNOWN_PREFIX}1"
        result = d.identify_embedding(_vec(0, 0, 0.7, 0.714), 2.0, 1.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"
        assert not result.confident
        assert d._pending == []

    def test_candidate_between_enrolled_speakers_is_not_promoted(self, monkeypatch):
        """登録話者たちの中間にいる候補は「混合」であって人ではない（aa 実測 08-26・5 回目）。"""
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer(
            DiarizerConfig(
                similarity_threshold=0.99,
                min_unknown_sec=0.0,
                min_unknown_utterances=1,
                mixture_similarity=0.73,
            )
        )
        # 実データでは登録話者同士も直交しない（自分 vs 相手A 0.788）。直交ベクトル 2 本だと
        # 両方に 0.73 以上似る点は存在しない（最大 0.707）ので、ボブはアリスに 0.8 似せる。
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", _vec(0.8, 0.6, 0, 0), SR)
        # アリス 0.95 / ボブ 0.94 → 両者の中間 → 起こさない
        between = _vec(0.95, 0.3, 0, 0)
        result = d.identify_embedding(between, 5.0, 0.0)
        assert result.name == f"{UNKNOWN_PREFIX}?"
        assert d.unknown_names == []
        assert d.mixture_rejections == 1
        assert d._pending == []
        # 両者から離れている（本物の 3 人目）→ 起こす
        result = d.identify_embedding(CAROL, 5.0, 1.0)
        assert result.is_new
        assert d.unknown_names == [f"{UNKNOWN_PREFIX}1"]

    def test_promoted_unknown_that_drifts_between_enrolled_is_dissolved(self, monkeypatch):
        """昇格後に両者の中間へ寄った不明話者は解体し、その後の発話は 不明話者? に落ちる。"""
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer(
            DiarizerConfig(
                similarity_threshold=0.99,
                min_unknown_sec=0.0,
                min_unknown_utterances=1,
                mixture_similarity=0.73,
                adapt_margin=0.0,
                enroll_weight=0.0,   # adapt 分だけで centroid が決まる＝寄りを再現しやすくする
            )
        )
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", _vec(0.8, 0.6, 0, 0), SR)
        # 両者から離れた本物として昇格
        first = d.identify_embedding(CAROL, 5.0, 0.0)
        assert first.is_new and d.unknown_names == [f"{UNKNOWN_PREFIX}1"]
        # 以後、両者の中間に近い embedding を吸って centroid が寄る（不明話者1 に 0.99 未満で最近傍）
        between = _vec(0.95, 0.3, 0.3, 0)
        d._speakers[f"{UNKNOWN_PREFIX}1"].adapt_embeddings.extend([between] * 10)
        result = d.identify_embedding(CAROL, 5.0, 1.0)
        assert f"{UNKNOWN_PREFIX}1" in d.dissolved_names
        assert d.unknown_names == [] or d.unknown_names == [f"{UNKNOWN_PREFIX}2"]
        assert result.name != f"{UNKNOWN_PREFIX}1"

    def test_substantive_unknown_is_protected_from_dissolution(self, monkeypatch):
        """7 回目: 本物の 3 人目が解体閾値まで 0.0094。文字/発話が人の量なら解体しない。"""
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer(
            DiarizerConfig(
                similarity_threshold=0.99,
                min_unknown_sec=0.0,
                min_unknown_utterances=1,
                mixture_similarity=0.73,
                adapt_margin=0.0,
                enroll_weight=0.0,
                dissolve_min_chars_per_utterance=10.0,
                dissolve_min_utterances=5,
            )
        )
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", _vec(0.8, 0.6, 0, 0), SR)
        d.identify_embedding(CAROL, 5.0, 0.0)
        name = f"{UNKNOWN_PREFIX}1"
        for _ in range(6):
            d.report_text(name, 21)   # 21 文字/発話 = 人
        between = _vec(0.95, 0.3, 0.3, 0)
        d._speakers[name].adapt_embeddings.extend([between] * 10)
        d.identify_embedding(CAROL, 5.0, 1.0)
        assert name in d.unknown_names
        assert d.dissolved_names == []
        # 断片（5 文字/発話）なら解体される
        d2 = EnrolledDiarizer(d.config)
        d2.enroll("アリス", ALICE, SR)
        d2.enroll("ボブ", _vec(0.8, 0.6, 0, 0), SR)
        d2.identify_embedding(CAROL, 5.0, 0.0)
        for _ in range(6):
            d2.report_text(name, 5)
        d2._speakers[name].adapt_embeddings.extend([between] * 10)
        d2.identify_embedding(CAROL, 5.0, 1.0)
        assert name in d2.dissolved_names

    def test_identify_result_is_json_serializable(self, monkeypatch):
        """numpy.bool / numpy.float が JSON の境界に漏れない（aa 実機検査 4 回目の回帰）。"""
        import json
        from dataclasses import asdict

        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer(DiarizerConfig(similarity_threshold=0.5, min_pool_sec=3.0))
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", BOB, SR)
        for duration in (5.0, 1.0):
            result = d.identify_embedding(ALICE_AGAIN, duration, 0.0)
            payload = asdict(result)
            assert type(payload["adapted"]) is bool
            assert type(payload["similarity"]) is float
            json.dumps(payload)

    def test_identify_result_reports_whether_it_adapted(self, monkeypatch):
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer(DiarizerConfig(similarity_threshold=0.5, min_pool_sec=3.0))
        d.enroll("アリス", ALICE, SR)
        assert d.identify_embedding(ALICE_AGAIN, 5.0, 0.0).adapted
        assert not d.identify_embedding(ALICE_AGAIN, 1.0, 1.0).adapted

    def test_pairwise_similarity_and_version_one_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            EnrolledDiarizer,
            "embed",
            staticmethod(lambda audio, sample_rate: np.asarray(audio, dtype=np.float32)[:4]),
        )
        d = EnrolledDiarizer()
        d.enroll("アリス", ALICE, SR)
        d.enroll("ボブ", BOB, SR)
        assert d.pairwise_similarity() == {("アリス", "ボブ"): pytest.approx(0.0)}
        path = tmp_path / "version1.json"
        path.write_text(
            '{"version": 1, "similarity_threshold": 0.6, "min_new_speaker_sec": 3, "speakers": []}',
            encoding="utf-8",
        )
        assert EnrolledDiarizer.load(path).config.similarity_threshold == 0.6


class TestRename:
    def test_rename_unknown_to_real_name(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        result = diarizer.identify(long(CAROL), SR)

        diarizer.rename(result.name, "キャロル")
        assert "キャロル" in diarizer.speaker_names
        assert result.name not in diarizer.speaker_names
        assert diarizer.identify(long(CAROL), SR).name == "キャロル"

    def test_rename_merges_into_existing(self, diarizer):
        diarizer.enroll("アリス", ALICE, SR)
        result = diarizer.identify(long(CAROL), SR)
        diarizer.rename(result.name, "アリス")
        assert diarizer.speaker_names == ["アリス"]
        assert diarizer._speakers["アリス"].sample_count == 2

    def test_rename_missing_speaker_raises(self, diarizer):
        with pytest.raises(KeyError):
            diarizer.rename("いない人", "誰か")


class TestPersistence:
    def test_round_trip(self, diarizer, tmp_path, monkeypatch):
        diarizer.enroll("アリス", ALICE, SR)
        diarizer.enroll("ボブ", BOB, SR)
        diarizer.identify(long(CAROL), SR)  # 暫定話者を1人起こす

        path = tmp_path / "speakers.json"
        diarizer.save(path)
        assert path.exists()

        def fake_embed(audio, sample_rate):
            array = np.asarray(audio, dtype=np.float32)
            if array.size < 4:
                return None
            return array[:4] / np.linalg.norm(array[:4])

        monkeypatch.setattr(EnrolledDiarizer, "embed", staticmethod(fake_embed))
        restored = EnrolledDiarizer.load(path)

        assert restored.enrolled_names == ["アリス", "ボブ"]
        assert len(restored.unknown_names) == 1
        assert restored.identify(long(ALICE_AGAIN), SR).name == "アリス"

    def test_unknown_counter_survives_reload(self, diarizer, tmp_path, monkeypatch):
        """再起動後に『不明話者1』が2人できないこと。"""
        diarizer.enroll("アリス", ALICE, SR)
        first = diarizer.identify(long(CAROL), SR)

        path = tmp_path / "speakers.json"
        diarizer.save(path)

        def fake_embed(audio, sample_rate):
            array = np.asarray(audio, dtype=np.float32)
            return array[:4] / np.linalg.norm(array[:4])

        monkeypatch.setattr(EnrolledDiarizer, "embed", staticmethod(fake_embed))
        restored = EnrolledDiarizer.load(path)
        second = restored.identify(long(DAVE), SR)

        assert second.name != first.name
        assert second.name == f"{UNKNOWN_PREFIX}2"

    def test_save_creates_parent_dirs(self, diarizer, tmp_path):
        diarizer.enroll("アリス", ALICE, SR)
        path = tmp_path / "sessions" / "2026-09-08" / "speakers.json"
        diarizer.save(path)
        assert path.exists()


class TestAudioHelpers:
    def test_peak_normalize_scales_to_095(self):
        out = peak_normalize(np.array([0.1, -0.2, 0.05], dtype=np.float32))
        assert np.max(np.abs(out)) == pytest.approx(0.95, abs=1e-6)

    def test_peak_normalize_returns_none_on_silence(self):
        assert peak_normalize(np.zeros(100, dtype=np.float32)) is None

    def test_resample_changes_length(self):
        out = resample(np.zeros(48000, dtype=np.float32), 48000, 16000)
        assert len(out) == 16000

    def test_resample_noop_when_tiny(self):
        tiny = np.array([0.5], dtype=np.float32)
        assert len(resample(tiny, 48000, 16000)) == 1


class TestSubstantiveClustersDoNotMergeEasily:
    """2026-09-11 の本番（登録なしで流し直し）: 相手 2 人が会議の後半で 1 つにまとまった。

    断片の吸収は 0.75 のままでよいが、どちらも中身のある人どうしは 0.90 以上でないと併合しない。
    """

    def _diarizer(self) -> EnrolledDiarizer:
        return EnrolledDiarizer(
            DiarizerConfig(similarity_threshold=0.99, merge_threshold=0.75, merge_threshold_substantive=0.90,
                           min_unknown_sec=0.0, min_unknown_utterances=1, adapt=False,
                           dissolve_min_utterances=2, dissolve_min_chars_per_utterance=10.0)
        )

    def _two_clusters(self, d: EnrolledDiarizer) -> tuple[str, str]:
        """似た声で 2 つの不明話者クラスタを作る。"""
        first = d.identify_embedding(CAROL, 6.0, 0.0).name
        second = d.identify_embedding(_vec(0, 0, 0.8, 0.6), 6.0, 1.0).name
        assert first != second
        return first, second

    def _talk(self, d: EnrolledDiarizer, name: str, times: int, chars: int) -> None:
        for _ in range(times):
            d.report_text(name, chars)

    def test_two_talking_people_stay_apart(self):
        d = self._diarizer()
        first, second = self._two_clusters(d)
        for name in (first, second):
            self._talk(d, name, times=5, chars=40)   # どちらも中身のある人
        assert d.merge_unknowns() == []
        assert len(d.unknown_names) == 2

    def test_a_fragment_is_still_absorbed(self):
        """中身の無いクラスタは、これまでどおり 0.75 で吸収する。"""
        d = self._diarizer()
        first, second = self._two_clusters(d)
        self._talk(d, first, times=5, chars=40)
        self._talk(d, second, times=3, chars=2)   # 断片（短い発話ばかり）
        assert d.merge_unknowns()
        assert len(d.unknown_names) == 1

    def test_merge_waits_until_there_is_evidence(self):
        """昇格直後（発話が足りない）は、似ていても併合しない。"""
        d = self._diarizer()
        first, second = self._two_clusters(d)
        self._talk(d, first, times=1, chars=40)
        self._talk(d, second, times=1, chars=2)
        assert d.merge_unknowns() == []


def _speak(diarizer: EnrolledDiarizer, name: str, utterances: int, chars: int) -> None:
    """文字起こしの結果だけを溜める（声の判定は別テストで見ている）。"""
    for _ in range(utterances):
        diarizer.report_text(name, chars)


def _grow_unknown(diarizer: EnrolledDiarizer, vec: np.ndarray) -> None:
    """不明話者を1人起こす。"""
    diarizer.identify(long(vec), SR, at=0.0)


class TestFailedEnrollments:
    """2026-09-11 の本番で起きた形を検知する。

    登録した「参加者B」の声紋が本人の声とまったく一致せず（平均類似度 0.605）、
    実際は自動で育った `不明話者1` が本人を担っていた。画面には何も出なかった。
    """

    def test_登録が当たらず不明話者が担っていたら報告する(self, diarizer):
        diarizer.enroll("参加者A", ALICE, SR)
        diarizer.enroll("参加者B", CAROL, SR)
        _grow_unknown(diarizer, BOB)

        _speak(diarizer, "参加者A", utterances=40, chars=30)
        _speak(diarizer, "参加者B", utterances=2, chars=3)      # 相づちだけ拾った幻の話者
        _speak(diarizer, "不明話者1", utterances=60, chars=25)  # 実際の参加者Bさん

        failures = diarizer.failed_enrollments()
        assert [failure["name"] for failure in failures] == ["参加者B"]
        assert failures[0]["utterances"] == 2
        assert failures[0]["candidates"] == ["不明話者1"]

    def test_全員が喋っていれば何も言わない(self, diarizer):
        diarizer.enroll("参加者A", ALICE, SR)
        diarizer.enroll("参加者B", CAROL, SR)
        _grow_unknown(diarizer, BOB)

        _speak(diarizer, "参加者A", utterances=40, chars=30)
        _speak(diarizer, "参加者B", utterances=30, chars=25)
        _speak(diarizer, "不明話者1", utterances=60, chars=25)

        assert diarizer.failed_enrollments() == []

    def test_不明話者が育っていなければ言わない(self, diarizer):
        """静かな人が居るだけの会議で「登録が失敗した」と騒がない。"""
        diarizer.enroll("参加者A", ALICE, SR)
        diarizer.enroll("参加者B", CAROL, SR)

        _speak(diarizer, "参加者A", utterances=40, chars=30)
        _speak(diarizer, "参加者B", utterances=1, chars=4)

        assert diarizer.failed_enrollments() == []
