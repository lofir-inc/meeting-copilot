"""Common Voice から比較用の 1 組を作る（朗読データ。第一次選抜まで）。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import prep_common_voice as prep


def corpus(tmp_path: Path, rows: list[dict]) -> Path:
    source = tmp_path / "ja"
    (source / "clips").mkdir(parents=True)
    with (source / "validated.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "path", "sentence"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            (source / "clips" / row["path"]).write_bytes(b"mp3")
    return source


def test_validated_tsv_が無ければ言う(tmp_path):
    with pytest.raises(SystemExit, match="validated.tsv"):
        prep.rows_of(tmp_path)


def test_話者を散らす(tmp_path, monkeypatch):
    """同じ人ばかりだと、その声に強いエンジンが有利になる（1 人 3 本まで）。"""
    rows = [{"client_id": "a" * 20, "path": f"a{i}.mp3", "sentence": f"文{i}"} for i in range(10)]
    rows += [{"client_id": "b" * 20, "path": f"b{i}.mp3", "sentence": f"別{i}"} for i in range(10)]
    source = corpus(tmp_path, rows)
    monkeypatch.setattr(prep, "duration", lambda path: 5.0)

    chosen = prep.pick(prep.rows_of(source), source, wanted=10, seed=1)

    assert len(chosen) == 6                                  # 2 人 × 3 本
    assert len({row["speaker"] for row in chosen}) == 2


def test_短すぎる長すぎるクリップは使わない(tmp_path, monkeypatch):
    rows = [{"client_id": "a" * 20, "path": "short.mp3", "sentence": "短い"},
            {"client_id": "b" * 20, "path": "long.mp3", "sentence": "長い"},
            {"client_id": "c" * 20, "path": "ok.mp3", "sentence": "ちょうどよい"}]
    source = corpus(tmp_path, rows)
    monkeypatch.setattr(prep, "duration",
                        lambda path: {"short.mp3": 0.5, "long.mp3": 30.0}.get(Path(path).name, 5.0))

    chosen = prep.pick(prep.rows_of(source), source, wanted=10, seed=1)

    assert [row["sentence"] for row in chosen] == ["ちょうどよい"]


def test_同じ種なら同じ組み合わせ(tmp_path, monkeypatch):
    """比較をやり直したときに、同じ音声で測れること。"""
    rows = [{"client_id": f"{i:020d}", "path": f"{i}.mp3", "sentence": f"文{i}"} for i in range(30)]
    source = corpus(tmp_path, rows)
    monkeypatch.setattr(prep, "duration", lambda path: 5.0)

    first = prep.pick(prep.rows_of(source), source, wanted=5, seed=7)
    second = prep.pick(prep.rows_of(source), source, wanted=5, seed=7)

    assert [row["path"] for row in first] == [row["path"] for row in second]
