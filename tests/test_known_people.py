"""その会社の会議に出てくる人を集める（誤変換の候補の根拠にする）。

2026-09-18: ミナトの議事録に「社内（社長・北村氏）で検討」と出たが、ミナトに北村は居ない。
実際は「自分さん」（こちらの担当者）の聞き違いだった。
"""

from __future__ import annotations

import json

from src.known_people import collect, mentions, roster_block


class TestMentions:
    def test_敬称の付いた名前を拾う(self):
        found = mentions("自分さんと相談。社内（社長・北村氏）で検討。西川さんへ連絡。")

        assert found == {"自分", "北村", "西川"}

    def test_助詞を巻き込まない(self):
        """「件は西川さん」から「件は西川」を拾ってしまった（2026-09-18 に直した）。"""
        assert mentions("お客さんの件は西川さんへ。") == {"西川"}

    def test_人ではない語を拾わない(self):
        found = mentions("お客さんと皆さん。弊社の担当者さんと責任者さんから。")

        assert found == set()

    def test_話者ラベルの形も拾う(self):
        assert "MINATO南" in mentions("MINATO南専務と話した。")

    def test_空でも落ちない(self):
        assert mentions("") == set() and mentions(None) == set()


def _session(root, name, client, *, speakers=None, minutes=""):
    path = root / "sessions" / name
    path.mkdir(parents=True)
    (path / "client.json").write_text(json.dumps({"name": client}, ensure_ascii=False), encoding="utf-8")
    if speakers:
        (path / "transcripts.jsonl").write_text(
            "\n".join(json.dumps({"speaker": one, "text": "あ"}, ensure_ascii=False) for one in speakers),
            encoding="utf-8")
    if minutes:
        (path / "minutes.md").write_text(minutes, encoding="utf-8")
    return path


class TestCollect:
    def test_過去の会議の話者と議事録から集める(self, tmp_path):
        _session(tmp_path, "2026-09-15_0957", "株式会社ミナト",
                 speakers=["MINATO南専務", "不明話者1"],
                 minutes="西川さんが対応。自分さんと相談した。")

        found = collect(tmp_path / "sessions", "株式会社ミナト")

        assert "MINATO南専務" in found        # 名前を付けた話者
        assert "西川" in found and "自分" in found   # 話に上がった人（所属者に限らない）
        assert not any("不明話者" in one for one in found)

    def test_別の会社の会議は見ない(self, tmp_path):
        """会社をまたいで名前を混ぜない。"""
        _session(tmp_path, "a", "株式会社ミナト", minutes="西川さんが対応。")
        _session(tmp_path, "b", "株式会社GX", minutes="参加者Aさんが対応。")

        found = collect(tmp_path / "sessions", "株式会社ミナト")

        assert "西川" in found and "参加者A" not in found

    def test_渡された名前も混ぜる(self, tmp_path):
        """People Master のその会社の人など。"""
        found = collect(tmp_path / "sessions", "株式会社ミナト", extra=["南社長"])

        assert found == ["南社長"]

    def test_会議が一つも無くても落ちない(self, tmp_path):
        assert collect(tmp_path / "sessions", "株式会社ミナト") == []

    def test_声の登録の引き継ぎを名簿に入れない(self, tmp_path):
        """2026-09-18 実データ: speakers.json は会社をまたいで引き継がれ、
        さくら歯科の名簿に別会社の 参加者A が出た。実際に話した人だけを採る。"""
        path = _session(tmp_path, "a", "さくら歯科", speakers=["KDCさやか"])
        (path / "speakers.json").write_text(
            json.dumps({"people": [{"name": "参加者A"}]}, ensure_ascii=False), encoding="utf-8")

        found = collect(tmp_path / "sessions", "さくら歯科")

        assert "KDCさやか" in found and "参加者A" not in found


class TestRosterBlock:
    def test_一覧が空なら何も足さない(self):
        """無い前提で書かせない。"""
        assert roster_block([]) == ""

    def test_一覧に無い名前を候補に挙げさせる(self):
        told = roster_block(["南社長", "西川"])

        assert "南社長、西川" in told
        assert "誤変換の候補" in told

    def test_一覧に無いことを誤りと言わせない(self):
        """初めて話に出た人は必ず一覧に無い。疑うだけの道具にしない。"""
        told = roster_block(["西川"])

        assert "一覧に無い＝誤り、ではありません" in told
        assert "全文の表記のまま" in told
