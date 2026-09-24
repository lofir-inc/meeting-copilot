"""声の台帳 — 会議をまたいで「前にも出た人」を覚えておき、次の会議で**候補として出す**。

自動では名前を付けない。**候補を出して、人が 1 回押す**。理由は実測（2026-09-14）:

| | 類似度 |
|---|---|
| 同じ会議の中の本人 | 0.95〜0.99 |
| **別の会議の本人**（自分・08-26 → 09-03 の Zoom 録画） | **0.85〜0.92** |
| **別の会議の別人**（自分 と 参加者A） | **〜0.85** |

会議をまたぐと、本人と別人の類似度が**重なる**（マイク・回線・会議アプリが変わるため）。
閾値 1 本で自動に名前を付けると、別人に取引先の名前が付く。議事録で発言者を取り違えるほうが、
「不明話者1」のままより害が大きい。

押した／断ったは `voice_matches.jsonl` に残す。溜まれば「この類似度以上なら自動でよい」を
実データで決められる（使うほど精度が上がる、の入口）。

台帳は**手元にだけ**置く（`workspace/voices/`）。声紋は個人を識別できる情報なので、外へは出さない。
消すときは `scripts/voice_library.py forget <名前>`（ゴミ箱へ退避する）。
"""

from __future__ import annotations

import json
import logging
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
MATCHES_FILE = "voice_matches.jsonl"
"""候補を押した／断ったの記録（セッションごと）。自動で付けてよい類似度を、あとで実データから決める材料。"""


@dataclass
class VoiceLibraryConfig:
    """`settings.yaml` の `meeting.voice_library`。"""

    enabled: bool = False
    """既定は off。声紋は個人を識別できる情報なので、覚えるかは使う人が決める。"""

    path: str = "workspace/voices/library.json"
    suggest_floor: float = 0.80
    """これ未満の人は候補に出さない。実測で別人の上位が 0.80〜0.85 にいる＝**床であって、決め手ではない**。"""

    max_suggestions: int = 3
    min_samples: int = 2
    """不明話者の声がこの件数に育つまでは候補を出さない（1 発話の平均はぶれる）。"""

    min_row_sec: float = 3.0
    """会議のあとに覚えるとき、これより短い行は使わない（短い行の声紋はぶれる）。"""

    max_rows_per_person: int = 60
    min_rows_per_person: int = 5
    """これ未満しか話していない人は覚えない（声紋が安定しない）。"""

    consistency: float = 0.75
    """その人の行のうち、平均の声とこれ以上似ている行だけを使う（取り違えた行を落とす）。"""

    min_consistent_share: float = 0.6
    """上に残った行がこの割合を切ったら、**その名前に複数人が混ざっている**とみなして覚えない。"""

    max_meetings_per_person: int = 10

    @classmethod
    def from_mapping(cls, values: dict | None) -> "VoiceLibraryConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (values or {}).items() if key in known})


@dataclass
class Suggestion:
    name: str
    score: float
    last_session: str

    def to_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 3), "last_session": self.last_session}


@dataclass
class LearnResult:
    learned: dict[str, int] = field(default_factory=dict)
    """覚えた人 → 使った行の数。"""
    skipped: dict[str, str] = field(default_factory=dict)
    """覚えなかった人 → 理由。"""


class VoiceLibrary:
    """人ごとに、会議 1 本につき 1 つの「平均の声」を持つ（最大 `max_meetings_per_person` 本）。

    人の声を 1 つに平均しない。会議ごとにマイクや回線が違うので、**いちばん近い会議の声**で比べる。
    """

    def __init__(self, path: Path, config: VoiceLibraryConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config or VoiceLibraryConfig()
        self.people: dict[str, list[dict]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.people = {name: list(entries) for name, entries in (payload.get("people") or {}).items()}
        self._vectors: dict[str, list[np.ndarray]] = {}
        self._refresh()

    def __len__(self) -> int:
        return len(self.people)

    def _refresh(self) -> None:
        self._vectors = {
            name: [np.asarray(entry["centroid"], dtype=np.float32) for entry in entries]
            for name, entries in self.people.items()
        }

    # ------------------------------------------------------------------ 候補

    def suggest(self, embedding: np.ndarray, *, exclude: set[str] | None = None) -> list[Suggestion]:
        """この声に近い、過去の会議の人を返す（近い順・床以上だけ）。"""
        exclude = exclude or set()
        embedding = np.asarray(embedding, dtype=np.float32)
        found: list[Suggestion] = []
        for name, vectors in self._vectors.items():
            if name in exclude or not vectors:
                continue
            scores = [_cosine(embedding, vector) for vector in vectors]
            best = int(np.argmax(scores))
            if scores[best] >= self.config.suggest_floor:
                found.append(Suggestion(name, scores[best], self.people[name][best].get("session", "")))
        found.sort(key=lambda item: item.score, reverse=True)
        return found[: self.config.max_suggestions]

    # ------------------------------------------------------------------ 覚える

    def learn(self, session: str, voices: dict[str, list[np.ndarray]]) -> LearnResult:
        """会議 1 本ぶんの声を覚える（同じ会議を 2 回覚えたら、あとのほうで置き換える）。"""
        result = LearnResult()
        stamp = datetime.now(JST).isoformat(timespec="seconds")
        for name, embeddings in voices.items():
            if len(embeddings) < self.config.min_rows_per_person:
                result.skipped[name] = f"長い発言が {len(embeddings)} 件しかない"
                continue
            centroid = np.mean(embeddings, axis=0)
            kept = [e for e in embeddings if _cosine(e, centroid) >= self.config.consistency]
            share = len(kept) / len(embeddings)
            if share < self.config.min_consistent_share or len(kept) < self.config.min_rows_per_person:
                result.skipped[name] = f"声がそろわない（平均に近い行が {share:.0%}）。複数人が混ざっている可能性"
                continue
            centroid = np.mean(kept, axis=0)
            entries = [entry for entry in self.people.get(name, []) if entry.get("session") != session]
            entries.append({"session": session, "at": stamp, "rows": len(kept),
                            "centroid": [round(float(value), 6) for value in centroid]})
            self.people[name] = entries[-self.config.max_meetings_per_person:]
            result.learned[name] = len(kept)
        self._refresh()
        return result

    def rename(self, old: str, new: str) -> None:
        if old not in self.people:
            raise KeyError(f"台帳にいません: {old}")
        self.people.setdefault(new, [])
        self.people[new] = (self.people[new] + self.people.pop(old))[-self.config.max_meetings_per_person:]
        self._refresh()

    def forget(self, name: str, trash_dir: Path) -> Path:
        """台帳から外す。直接は消さない — ゴミ箱に退避してから外す（`99-trash/_RULE.md`）。"""
        if name not in self.people:
            raise KeyError(f"台帳にいません: {name}")
        trash_dir = Path(trash_dir)
        trash_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(JST).strftime("%Y%m%d-%H%M%S")
        target = trash_dir / f"voice-{stamp}-{_safe(name)}.json"
        target.write_text(json.dumps({"name": name, "entries": self.people[name]}, ensure_ascii=False),
                          encoding="utf-8")
        del self.people[name]
        self._refresh()
        return target

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "people": self.people}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
        try:
            self.path.chmod(0o600)   # 声紋は自分だけが読める状態にする
        except OSError:
            pass


def voices_from_session(
    rows: list[dict],
    audio: np.ndarray,
    sample_rate: int,
    *,
    self_name: str,
    config: VoiceLibraryConfig,
    embed,
) -> dict[str, list[np.ndarray]]:
    """会議の全文（話者つき）と相手側の録音から、人ごとの声紋を集める。

    自分（マイク側）と「不明話者」は覚えない。行は会議全体から均等に拾う（冒頭だけに寄せない）。
    """
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        name = str(row.get("speaker", ""))
        if not name or name == self_name or name.startswith("不明話者"):
            continue
        if float(row.get("end_time", 0)) - float(row.get("start_time", 0)) < config.min_row_sec:
            continue
        by_name.setdefault(name, []).append(row)
    voices: dict[str, list[np.ndarray]] = {}
    for name, picked in by_name.items():
        step = max(1, len(picked) // config.max_rows_per_person)
        embeddings = []
        for row in picked[::step][: config.max_rows_per_person]:
            clip = audio[int(float(row["start_time"]) * sample_rate): int(float(row["end_time"]) * sample_rate)]
            embedding = embed(clip, sample_rate)
            if embedding is not None:
                embeddings.append(np.asarray(embedding, dtype=np.float32))
        voices[name] = embeddings
    return voices


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """録音（16-bit PCM）を float32 のモノラルで読む。"""
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"16-bit 以外の録音は読めません: {path}（{width * 8} bit）")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def record_match(session_dir: Path, *, speaker: str, suggestion: dict, accepted: bool) -> None:
    """候補を押した／断ったを残す。"""
    entry = {"at": datetime.now(JST).isoformat(timespec="seconds"), "speaker": speaker,
             "name": suggestion.get("name"), "score": suggestion.get("score"),
             "last_session": suggestion.get("last_session", ""), "accepted": accepted}
    with (Path(session_dir) / MATCHES_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else -1.0


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)[:40] or "noname"
