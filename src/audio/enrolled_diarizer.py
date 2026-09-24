"""登録済み N 話者への最近傍照合による話者判定。

既存の `diarizer.py`（2話者固定・教師なしクラスタリング）とは別物。
あちらは「どちらにも似ていなければ、より近い方に分類」するため、**3人目が必ず
A か B に吸い込まれる**。閾値をいじっても直らない。

こちらは会議冒頭に「お名前と一言」を録って centroid を登録しておく前提に立つ。
embedding は「同じ人か」しか言えず「誰か」は言えないので、名前を先に紐付けることで
　　教師なしクラスタリング（脆い） → 登録済み N 人への最近傍照合（頑健）
に変わる。

閾値を下回った発話は握り潰さず `不明話者N` として**暫定登録**する。
同じ人が再登場すれば同じラベルが付くので、後から名寄せ（`rename()`）できる。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# resemblyzer は初回 import が重いので遅延ロード
_encoder = None

# resemblyzer が期待するサンプリングレート
ENCODER_SAMPLE_RATE = 16000

# これより短いチャンクは embedding が不安定なので判定しない
MIN_DURATION_SEC = 0.6

# 閾値を外れた発話は、すぐに新話者にせず候補プールで育てる。
#
# 実データ（2026-09-03 の1時間・2話者）で判明した問題:
#   登録話者に当たった発話は長さ中央値 10.0 秒／文字数中央値 42
#   不明話者になった発話は長さ中央値  1.4 秒／文字数中央値  0
# ＝ 不明話者の正体は相槌・息継ぎ・ノイズだった。議事録には1文字も寄与しないのに、
#    1件ごとに新しい話者を起こして登録簿を12人に膨らませ、speakers.json に
#    保存されて次回まで持ち越されていた。
# 候補は合計時間と発話数の両方を満たしてから昇格する。

UNKNOWN_PREFIX = "不明話者"


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder

        _encoder = VoiceEncoder()
        logger.info("Speaker encoder loaded")
    return _encoder


@dataclass
class DiarizerConfig:
    similarity_threshold: float = 0.75
    max_adapt_embeddings: int = 10
    adapt: bool = True
    min_unknown_sec: float = 6.0
    min_unknown_utterances: int = 2
    merge_threshold: float = 0.75
    merge_min_utterances: int = 3
    """併合の判断を始めるまでに、両方のクラスタに要る発話数。

    これが無いと、昇格した直後（まだ 1 発話）に 0.75 で併合され、別人が 1 人になる。
    2026-09-11 の本番を流し直すと、相手 2 人は会議の開始 1 分で 1 つにまとまっていた。
    """
    merge_threshold_substantive: float = 0.90
    """どちらも「中身のある人」の不明話者どうしを併合するのに要る類似度。

    断片を吸収するのは 0.75 のままでよいが、人と人を 0.75 で併合すると別人が 1 人になる。
    実測（2026-09-11 の本番を登録なしで流し直し）: 相手 2 人が会議の後半で 1 つにまとまり、
    話者の一致が 84% まで落ちた（15 分の抜粋では分かれて 92%）。
    """
    pending_ttl_sec: float = 600.0
    adapt_margin: float = 0.10
    enroll_weight: float = 0.5
    enroll_warn_similarity: float = 0.80
    mixture_similarity: float = 0.73
    """昇格候補の centroid が、登録済み話者 `mixture_min_speakers` 名以上とこの値以上に似ていたら
    「人」ではなく「混合（同時発話・断片・幻覚）」とみなして起こさない。

    aa の実測（08-26・5 回目）: 本物の 3 人目は 自分 0.708 / 相手A 0.687、断片クラスタは
    0.755 / 0.739 と**両者の中間**にいた。登録話者たちの中間にいるクラスタは人ではない。
    """
    mixture_min_speakers: int = 2
    dissolve_min_chars_per_utterance: float = 10.0
    """解体から守る「中身のある人」の下限（文字/発話）。

    埋め込みの距離 1 本で「人か混合か」を決めると薄氷になる（7 回目: 本物の 3 人目が解体閾値まで
    0.0094）。本物は 21.2 文字/発話、断片は 5.6 と 3.8 倍の開きがあるので、`report_text()` で
    溜めた文字数が `dissolve_min_utterances` 発話以上でこの値以上なら、0.73 を超えても解体しない。
    """
    dissolve_min_utterances: int = 5
    min_pool_sec: float = 3.0
    """候補プールと適応へ入れる run の最小長。"""
    short_turn_threshold: float = 0.65
    """短い run を既知話者へ暫定付与する最低類似度（保険。効いているのは下の margin）。"""
    short_turn_margin: float = 0.10
    """短い run を付与するのに必要な 1 位と 2 位の差。

    絶対値の床だけでは危ない: 別人 2 名の centroid が会議終了時に 0.672 まで並ぶ実測
    （2026-09-08 #10）があり、0.65 の床だけだと小島の断片が自分に付く。
    「A に近い」ではなく「A の方が B より明確に近い」で付ける。
    aa の分布（09-03・短い run 559 件）: margin 中央 0.231・25% 0.167。0.10 は裾の下で、
    素直な run はほぼ残しつつ 0.008 差のような泥だけを落とす位置。
    """


# eq=False: numpy 配列のリストを持つので、既定の __eq__ だと list.remove / in / == が
#   「配列の真偽値は曖昧」で落ちる（aa の実機検査 2026-09-08 で 3 人目の昇格時に発生）。同一性で比べる。
@dataclass(eq=False)
class PendingCluster:
    """昇格前の未登録話者候補。"""

    embeddings: list[np.ndarray] = field(default_factory=list)
    total_sec: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0

    def centroid(self) -> np.ndarray:
        """候補に入った embedding の平均を返す。"""
        return np.mean(self.embeddings, axis=0)


@dataclass(eq=False)  # 同上。embedding のリストを持つので同一性で比べる
class Speaker:
    """1話者分の登録情報。"""

    name: str
    enrolled: bool = True
    """True = 冒頭の登録フローで名前つきで登録された。False = 実行中に現れた暫定話者。"""

    enroll_embeddings: list[np.ndarray] = field(default_factory=list)
    """登録時の embedding。これは**捨てない**（本人の錨）。"""

    adapt_embeddings: list[np.ndarray] = field(default_factory=list)
    """実行中に追加された embedding。直近 N 件のみ保持する。"""

    def centroid(self, max_adapt: int, enroll_weight: float) -> np.ndarray:
        """登録分を錨にして直近 adapt 分を重み付きで混ぜる。"""
        enrolled = np.mean(self.enroll_embeddings, axis=0)
        adapted = self.adapt_embeddings[-max_adapt:]
        if not adapted:
            return enrolled
        adaptation = np.mean(adapted, axis=0)
        return enroll_weight * enrolled + (1 - enroll_weight) * adaptation

    @property
    def sample_count(self) -> int:
        return len(self.enroll_embeddings) + len(self.adapt_embeddings)


@dataclass
class IdentifyResult:
    """話者判定の結果。"""

    name: str
    similarity: float
    is_new: bool = False
    """新しく暫定話者を起こしたか。"""

    confident: bool = True
    """閾値を超えて照合できたか。False なら暫定話者にフォールバックしている。"""

    adapted: bool = False
    """この embedding を話者の centroid に取り込んだか（声紋接近の分析用）。"""


Partial = tuple[float, float, np.ndarray]


class EnrolledDiarizer:
    """登録済み N 話者への最近傍照合器。"""

    def __init__(self, config: DiarizerConfig | None = None, **legacy: object) -> None:
        """設定と旧キーワード引数の両方から照合器を作成する。

        旧キーワード引数（similarity_threshold= / max_adapt_embeddings= / adapt=）は
        DiarizerConfig の同名フィールドを上書きするだけ。候補プールの条件は変えない
        （旧引数で作ったときだけ即時昇格に戻るような二重仕様は作らない）。
        """
        values = asdict(config) if config is not None else asdict(DiarizerConfig())
        for name in ("similarity_threshold", "max_adapt_embeddings", "adapt"):
            if name in legacy:
                values[name] = legacy.pop(name)
        if "min_new_speaker_sec" in legacy:
            legacy.pop("min_new_speaker_sec")
            logger.warning("min_new_speaker_sec は候補プール方式では無視されます")
        if legacy:
            unexpected = next(iter(legacy))
            raise TypeError(f"予期しないキーワード引数: {unexpected}")
        self.config = DiarizerConfig(**values)
        self._threshold = self.config.similarity_threshold
        self._max_adapt = self.config.max_adapt_embeddings
        self._adapt = self.config.adapt
        self._speakers: dict[str, Speaker] = {}
        self._pending: list[PendingCluster] = []
        self._unknown_count = 0
        self.mixture_rejections = 0
        """混合として捨てた昇格候補の数（評価用）。"""
        self.dissolved_names: list[str] = []
        """昇格後に混合と判明して解体した不明話者のラベル（字幕の付け直し用）。"""
        self._text_stats: dict[str, list[int]] = {}
        """話者名 → [発話数, 文字数]。report_text() で溜め、解体の判定に使う。"""

    # ------------------------------------------------------------------ 登録

    def enroll(self, name: str, audio: np.ndarray, sample_rate: int) -> None:
        """名前つきで話者を登録する。同名で複数回呼べば embedding が足される。"""
        name = name.strip()
        if not name:
            raise ValueError("話者名が空です。")
        embedding = self.embed(audio, sample_rate)
        if embedding is None:
            raise ValueError(
                f"{name}: 音声が短すぎるか無音です（{MIN_DURATION_SEC} 秒以上の発話が必要）。"
            )
        speaker = self._speakers.get(name)
        if speaker is None:
            speaker = Speaker(name=name, enrolled=True)
            self._speakers[name] = speaker
        else:
            # 暫定話者に名前を与えた形にもなる
            speaker.enrolled = True
        speaker.enroll_embeddings.append(embedding)
        logger.info("話者を登録: %s (embeddings=%d)", name, speaker.sample_count)

    def rename(self, old_name: str, new_name: str) -> None:
        """話者ラベルを付け替える（`不明話者1` → 実名 の名寄せ用）。"""
        if old_name not in self._speakers:
            raise KeyError(f"未知の話者: {old_name}")
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("新しい話者名が空です。")
        speaker = self._speakers.pop(old_name)
        speaker.name = new_name
        existing = self._speakers.get(new_name)
        if existing is None:
            self._speakers[new_name] = speaker
        else:
            # 併合
            existing.enroll_embeddings.extend(speaker.enroll_embeddings)
            existing.adapt_embeddings.extend(speaker.adapt_embeddings)
        logger.info("話者を名寄せ: %s → %s", old_name, new_name)

    def add_anchor(self, name: str, embedding: np.ndarray) -> None:
        """人が名前を付けた発話の embedding を、その話者の錨として足す。

        会議の途中で UI から「この行は 参加者A」と直したときに呼ぶ。冒頭の登録と同じ扱いにするので、
        その後の短い発話もこの話者へ付くようになる（短い run は登録済み話者にしか付けないため）。
        """
        name = name.strip()
        if not name:
            raise ValueError("話者名が空です。")
        speaker = self._speakers.get(name)
        if speaker is None:
            speaker = Speaker(name=name, enrolled=True)
            self._speakers[name] = speaker
        speaker.enrolled = True
        speaker.enroll_embeddings.append(np.asarray(embedding, dtype=np.float32))
        logger.info("発話から話者の錨を追加: %s (embeddings=%d)", name, speaker.sample_count)

    def mark_named(self, name: str) -> None:
        """人が付けた名前の話者を登録済みとして扱う（UI の改名で呼ぶ）。

        暫定話者（enrolled=False）のままだと、改名しても短い発話が付かず「不明話者?」が残る。
        内部の併合（不明話者2 → 不明話者1）では呼ばない。
        """
        speaker = self._speakers.get(name)
        if speaker is not None and not name.startswith("不明話者"):
            speaker.enrolled = True

    # ------------------------------------------------------------------ 判定

    def identify(self, audio: np.ndarray, sample_rate: int, at: float = 0.0) -> IdentifyResult:
        """音声チャンクの話者名を返す。"""
        embedding = self.embed(audio, sample_rate)
        if embedding is None:
            return IdentifyResult(name=self._unknown_label(), similarity=-1.0, confident=False)
        duration = np.asarray(audio).size / sample_rate
        return self.identify_embedding(embedding, duration, at)

    def identify_embedding(
        self,
        embedding: np.ndarray,
        duration_sec: float,
        at: float,
    ) -> IdentifyResult:
        """既計算の embedding を登録話者または候補プールへ照合する。"""
        self._expire_pending(at)
        embedding = np.asarray(embedding, dtype=np.float32)
        scores = self.scores(embedding)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if ordered and ordered[0][1] >= self._threshold:
            best_name, best_score = ordered[0]
            margin = best_score - ordered[1][1] if len(ordered) > 1 else float("inf")
            # bool() で包む: margin は numpy float なので比較結果が numpy.bool になり、
            #   JSON 化で「Object of type bool is not JSON serializable」と落ちる（aa 実測 4 回目）。
            adapted = bool(
                self._adapt
                and duration_sec >= self.config.min_pool_sec
                and margin >= self.config.adapt_margin
            )
            if adapted:
                self._speakers[best_name].adapt_embeddings.append(embedding)
            logger.debug(
                "話者類似度: %s (threshold=%.2f)",
                ", ".join(f"{name}={score:.3f}" for name, score in ordered),
                self._threshold,
            )
            return IdentifyResult(
                name=best_name, similarity=float(best_score), confident=True, adapted=adapted
            )

        if duration_sec < self.config.min_pool_sec:
            # 短い run は候補プールに入れない（断片から偽クラスタが育つのを防ぐ）。
            #   付けるのは「1 位が床以上」かつ「1 位と 2 位の差が margin 以上」のときだけ。
            #   照合先は**登録済み話者だけ**。昇格済みの不明話者に当てると、断片が不明クラスタを
            #   吸い寄せて太らせる（08-26 の 不明話者7＝188 発話・中央 2.0 秒・幻覚まじり。
            #   aa 実測 2026-09-08 16:50）。本物の 3 人目は 3 秒以上の run で昇格するので失われない。
            enrolled_only = [
                (name, score) for name, score in ordered if self._speakers[name].enrolled
            ]
            best_name, best_score = (
                enrolled_only[0] if enrolled_only else (self._unknown_label(), -1.0)
            )
            margin = (
                best_score - enrolled_only[1][1] if len(enrolled_only) > 1 else float("inf")
            )
            if (
                best_score >= self.config.short_turn_threshold
                and margin >= self.config.short_turn_margin
            ):
                logger.debug(
                    "短い run を既知話者へ暫定付与: %s=%.3f margin=%.3f (床=%.2f, margin≥%.2f)",
                    best_name,
                    best_score,
                    margin,
                    self.config.short_turn_threshold,
                    self.config.short_turn_margin,
                )
                return IdentifyResult(name=best_name, similarity=best_score, confident=False)
            logger.debug(
                "短い run は判定保留: best=%.3f margin=%.3f (床=%.2f, margin≥%.2f)",
                best_score,
                margin,
                self.config.short_turn_threshold,
                self.config.short_turn_margin,
            )
            return IdentifyResult(name=self._unknown_label(), similarity=best_score, confident=False)

        pending = self._find_pending(embedding)
        if pending is None:
            pending = PendingCluster(
                embeddings=[embedding],
                total_sec=duration_sec,
                first_seen=at,
                last_seen=at,
            )
            self._pending.append(pending)
        else:
            pending.embeddings.append(embedding)
            pending.total_sec += duration_sec
            pending.last_seen = at
        result = IdentifyResult(
            name=self._unknown_label(),
            similarity=ordered[0][1] if ordered else -1.0,
            confident=False,
        )
        if (
            pending.total_sec >= self.config.min_unknown_sec
            and len(pending.embeddings) >= self.config.min_unknown_utterances
        ):
            if self._looks_like_mixture(pending):
                # 登録話者たちの中間にいる候補は人ではない。起こさずに候補ごと捨てる
                self._pending.remove(pending)
                self.mixture_rejections += 1
                logger.info(
                    "昇格候補を混合として破棄 (%d 件目): 登録話者との類似度 %s",
                    self.mixture_rejections,
                    {n: round(v, 3) for n, v in self._enrolled_similarity(pending.centroid()).items()},
                )
                merged = self.merge_unknowns()
                if merged:
                    logger.info("不明話者クラスタを併合: %s", merged)
                return result
            new_name = self._new_unknown()
            self._speakers[new_name] = Speaker(
                name=new_name,
                enrolled=False,
                enroll_embeddings=pending.embeddings,
            )
            self._pending.remove(pending)
            result = IdentifyResult(
                name=new_name,
                similarity=result.similarity,
                is_new=True,
                confident=False,
            )
            logger.info("未登録の話者を検出: %s", new_name)
        merged = self.merge_unknowns()
        if merged:
            logger.info("不明話者クラスタを併合: %s", merged)
        dissolved = self.dissolve_mixtures()
        if dissolved:
            logger.info("混合と判明した不明話者を解体: %s", dissolved)
            if result.name in dissolved:
                result = IdentifyResult(
                    name=self._unknown_label(), similarity=result.similarity, confident=False
                )
        return result

    def report_text(self, name: str, chars: int) -> None:
        """文字起こしの結果（話者名と文字数）を溜める。解体の判定材料。"""
        stats = self._text_stats.setdefault(name, [0, 0])
        stats[0] += 1
        stats[1] += chars

    def chars_per_utterance(self, name: str) -> tuple[int, float]:
        """(発話数, 文字/発話) を返す。未報告なら (0, 0.0)。"""
        utterances, chars = self._text_stats.get(name, [0, 0])
        return utterances, (chars / utterances if utterances else 0.0)

    def _is_substantive(self, name: str) -> bool:
        """文字起こしの中身が「人」の量か（断片の寄せ集めは 1 発話あたりの文字が桁で少ない）。"""
        utterances, per_utterance = self.chars_per_utterance(name)
        return (
            utterances >= self.config.dissolve_min_utterances
            and per_utterance >= self.config.dissolve_min_chars_per_utterance
        )

    def failed_enrollments(self) -> list[dict]:
        """登録したのに、この会議で一度も当たらなかった話者を洗い出す。

        2026-09-11 の本番で実際に起きていた（誰も気づかないまま会議が終わった）。
        登録した「参加者B」の声紋は本人の声と一致せず（本人の発話 73 件との平均類似度 0.605）、
        実際に担っていたのは自動で育った `不明話者1` だった（0.861・67/73 件）。
        話者の一致率 93% が出たのは**不明話者プールの回収が働いたから**であって、登録が効いた
        からではない。副作用として、幻の登録話者が相づち 219 件を吸い、議事録に別人として載る。

        「登録済みなのに中身のある発話が無い」かつ「中身のある不明話者が育っている」＝ 登録の失敗。
        判定に声紋の距離は使わない（失敗した登録は本人の声と似ていないので、距離では気づけない）。
        """
        unknown = [name for name in self.unknown_names if self._is_substantive(name)]
        if not unknown:
            return []
        failures: list[dict] = []
        for name in self.enrolled_names:
            if self._is_substantive(name):
                continue
            utterances, per_utterance = self.chars_per_utterance(name)
            failures.append({
                "name": name,
                "utterances": utterances,
                "chars_per_utterance": round(per_utterance, 1),
                "candidates": list(unknown),
            })
        return failures

    def dissolve_mixtures(self) -> list[str]:
        """昇格後に登録話者たちの中間へ寄ってしまった不明話者を解体する。

        昇格時のゲートは 1 回しか見ない。昇格直後は中間に居なくても、その後に断片を吸って
        両者の中間へ寄る（08-26 の 不明話者7: 昇格後に 0.767 / 0.773。aa 実測 6 回目）。
        解体したラベルは `dissolved_names` に残し、呼び出し側が字幕を `不明話者?` に付け直す。
        """
        dissolved: list[str] = []
        for name in list(self.unknown_names):
            speaker = self._speakers[name]
            centroid = speaker.centroid(self._max_adapt, self.config.enroll_weight)
            close = [
                value
                for value in self._enrolled_similarity(centroid).values()
                if value >= self.config.mixture_similarity
            ]
            if len(close) >= self.config.mixture_min_speakers:
                if self._is_substantive(name):
                    logger.debug(
                        "解体候補だが中身があるので守る: %s (発話 %d・%.1f 文字/発話)",
                        name, *self.chars_per_utterance(name),
                    )
                    continue
                self._speakers.pop(name)
                self.dissolved_names.append(name)
                self.mixture_rejections += 1
                dissolved.append(name)
        return dissolved

    def partial_embeddings(
        self,
        audio: np.ndarray,
        sample_rate: int,
        rate: float = 2.5,
    ) -> list[Partial]:
        """resemblyzer の部分 embedding と実際の slice 時刻を返す。"""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != ENCODER_SAMPLE_RATE:
            audio = resample(audio, sample_rate, ENCODER_SAMPLE_RATE)
        audio = peak_normalize(audio)
        if audio is None:
            return []
        _, partials, slices = _get_encoder().embed_utterance(
            audio,
            return_partials=True,
            rate=rate,
        )
        return [
            (
                audio_slice.start / ENCODER_SAMPLE_RATE,
                audio_slice.stop / ENCODER_SAMPLE_RATE,
                embedding,
            )
            for embedding, audio_slice in zip(partials, slices)
        ]

    def merge_unknowns(self) -> list[tuple[str, str]]:
        """似た不明話者クラスタだけを rename() 経由で併合する。"""
        merged: list[tuple[str, str]] = []
        names = list(self.unknown_names)
        for index, old_name in enumerate(names):
            if old_name not in self._speakers:
                continue
            for new_name in names[index + 1 :]:
                if new_name not in self._speakers:
                    continue
                old = self._speakers[old_name]
                new = self._speakers[new_name]
                similarity = _cosine(
                    old.centroid(self._max_adapt, self.config.enroll_weight),
                    new.centroid(self._max_adapt, self.config.enroll_weight),
                )
                # どちらも中身のある人なら、別人が似てきただけの可能性が高いので慎重に併合する。
                #   2026-09-11 の本番（登録なしで流し直し）では、相手の 2 人が会議の後半で 1 つにまとまった
                #   （15 分の抜粋では分かれていた＝適応で centroid が寄っていく）。断片の吸収は今までどおり。
                seen = min(self.chars_per_utterance(old_name)[0], self.chars_per_utterance(new_name)[0])
                if seen < self.config.merge_min_utterances:
                    # まだ「人か断片か」を決める材料が無い。数発話ぶん待つ
                    continue
                both_substantive = self._is_substantive(old_name) and self._is_substantive(new_name)
                threshold = self.config.merge_threshold_substantive if both_substantive else self.config.merge_threshold
                if similarity >= threshold:
                    # 先に起きた方のラベルを残す（字幕には既にそのラベルで出ているため）
                    self.rename(new_name, old_name)
                    merged.append((new_name, old_name))
        return merged

    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        """各登録話者との コサイン類似度を返す。"""
        return {
            name: _cosine(embedding, speaker.centroid(self._max_adapt, self.config.enroll_weight))
            for name, speaker in self._speakers.items()
        }

    def voice_of(self, name: str) -> tuple[np.ndarray, int] | None:
        """その話者のいまの声（centroid）と、声の件数を返す。声の台帳の候補づくりに使う。"""
        speaker = self._speakers.get(name)
        if speaker is None or not speaker.enroll_embeddings:
            return None
        return speaker.centroid(self._max_adapt, self.config.enroll_weight), speaker.sample_count

    def pairwise_similarity(self, enrolled_only: bool = True) -> dict[tuple[str, str], float]:
        """話者 centroid の総当たりコサイン類似度を返す。"""
        names = self.enrolled_names if enrolled_only else self.speaker_names
        return {
            (left, right): _cosine(
                self._speakers[left].centroid(self._max_adapt, self.config.enroll_weight),
                self._speakers[right].centroid(self._max_adapt, self.config.enroll_weight),
            )
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        }

    def unknown_vs_enrolled(self) -> dict[str, dict[str, float]]:
        """各不明話者 centroid と登録話者 centroid の類似度を返す。"""
        similarities = self.pairwise_similarity(enrolled_only=False)
        return {
            unknown: {
                enrolled: similarities[(enrolled, unknown)]
                if (enrolled, unknown) in similarities
                else similarities[(unknown, enrolled)]
                for enrolled in self.enrolled_names
            }
            for unknown in self.unknown_names
        }

    def self_drift(self) -> dict[str, float]:
        """登録時 centroid と現在 centroid の類似度を話者ごとに返す。"""
        return {
            name: _cosine(
                np.mean(speaker.enroll_embeddings, axis=0),
                speaker.centroid(self._max_adapt, self.config.enroll_weight),
            )
            for name, speaker in self._speakers.items()
            if speaker.enroll_embeddings
        }

    # ------------------------------------------------------------------ 情報

    @property
    def speaker_names(self) -> list[str]:
        return list(self._speakers)

    @property
    def enrolled_names(self) -> list[str]:
        return [name for name, speaker in self._speakers.items() if speaker.enrolled]

    @property
    def unknown_names(self) -> list[str]:
        return [name for name, speaker in self._speakers.items() if not speaker.enrolled]

    def __len__(self) -> int:
        return len(self._speakers)

    # ------------------------------------------------------------ 永続化

    def save(self, path: str | Path) -> None:
        """`speakers.json` に保存する（再起動しても登録し直さなくていい）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "config": asdict(self.config),
            "unknown_count": self._unknown_count,
            "speakers": [
                {
                    "name": speaker.name,
                    "enrolled": speaker.enrolled,
                    "enroll_embeddings": [
                        embedding.tolist() for embedding in speaker.enroll_embeddings
                    ],
                    "adapt_embeddings": [
                        embedding.tolist()
                        for embedding in speaker.adapt_embeddings[-self._max_adapt :]
                    ],
                }
                for speaker in self._speakers.values()
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("話者登録を保存: %s (%d 名)", path, len(self._speakers))

    @classmethod
    def load(
        cls,
        path: str | Path,
        similarity_threshold: float | None = None,
        max_adapt_embeddings: int = 10,
        adapt: bool = True,
        **legacy: object,
    ) -> "EnrolledDiarizer":
        """`speakers.json` から復元する。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = asdict(DiarizerConfig())
        if payload.get("version", 1) >= 2:
            known = set(DiarizerConfig.__dataclass_fields__)
            values.update(
                {
                    name: value
                    for name, value in payload.get("config", {}).items()
                    if name in known
                }
            )
        else:
            values["similarity_threshold"] = payload.get("similarity_threshold", 0.75)
        if similarity_threshold is not None:
            values["similarity_threshold"] = similarity_threshold
        values["max_adapt_embeddings"] = max_adapt_embeddings
        values["adapt"] = adapt
        diarizer = cls(DiarizerConfig(**values), **legacy)
        diarizer._unknown_count = int(payload.get("unknown_count", 0))
        for entry in payload.get("speakers", []):
            speaker = Speaker(
                name=entry["name"],
                enrolled=bool(entry.get("enrolled", True)),
                enroll_embeddings=[
                    np.asarray(embedding, dtype=np.float32)
                    for embedding in entry.get("enroll_embeddings", [])
                ],
                adapt_embeddings=[
                    np.asarray(embedding, dtype=np.float32)
                    for embedding in entry.get("adapt_embeddings", [])
                ],
            )
            diarizer._speakers[speaker.name] = speaker
        logger.info("話者登録を読み込み: %s (%d 名)", path, len(diarizer._speakers))
        return diarizer

    # ------------------------------------------------------------ 内部

    def _find_pending(self, embedding: np.ndarray) -> PendingCluster | None:
        """閾値以上に似た候補プールを返す。"""
        matches = [
            (pending, _cosine(embedding, pending.centroid()))
            for pending in self._pending
        ]
        if not matches:
            return None
        pending, similarity = max(matches, key=lambda item: item[1])
        return pending if similarity >= self.config.merge_threshold else None

    def _enrolled_similarity(self, embedding: np.ndarray) -> dict[str, float]:
        """登録済み話者だけとの cos を返す。"""
        return {
            name: _cosine(embedding, speaker.centroid(self._max_adapt, self.config.enroll_weight))
            for name, speaker in self._speakers.items()
            if speaker.enrolled
        }

    def _looks_like_mixture(self, pending: PendingCluster) -> bool:
        """候補の centroid が登録話者 N 名以上と mixture_similarity 以上に似ているか。"""
        close = [
            value
            for value in self._enrolled_similarity(pending.centroid()).values()
            if value >= self.config.mixture_similarity
        ]
        return len(close) >= self.config.mixture_min_speakers

    def _expire_pending(self, at: float) -> None:
        """最後の観測から TTL を超えた候補を捨てる。"""
        before = len(self._pending)
        self._pending = [
            pending
            for pending in self._pending
            if at - pending.last_seen <= self.config.pending_ttl_sec
        ]
        if len(self._pending) != before:
            logger.debug(
                "期限切れの未登録話者候補を破棄: %d 件",
                before - len(self._pending),
            )

    def _new_unknown(self) -> str:
        self._unknown_count += 1
        return f"{UNKNOWN_PREFIX}{self._unknown_count}"

    def _unknown_label(self) -> str:
        """判定不能（候補育成中等）のときのラベル。話者は起こさない。"""
        return f"{UNKNOWN_PREFIX}?"

    @staticmethod
    def embed(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """音声チャンクから speaker embedding を計算する。

        短すぎる・無音の場合は None。
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size / sample_rate < MIN_DURATION_SEC:
            return None
        if sample_rate != ENCODER_SAMPLE_RATE:
            audio = resample(audio, sample_rate, ENCODER_SAMPLE_RATE)
        audio = peak_normalize(audio)
        if audio is None:
            return None
        return _get_encoder().embed_utterance(audio)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def peak_normalize(audio: np.ndarray) -> np.ndarray | None:
    """ピーク正規化 — 最大振幅を 0.95 に。完全な無音なら None。"""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return None
    return (audio / peak * 0.95).astype(np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """簡易リサンプリング（線形補間）。"""
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    if target_len <= 1 or len(audio) <= 1:
        return audio.astype(np.float32)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
