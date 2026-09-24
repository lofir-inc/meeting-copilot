"""窓の切れ目で割れた文をつなぎ直す（2026-09-18 の実会議で出た形）。"""

from __future__ import annotations

from src.text.stitch import pull_back, stitch


class TestPullBack:
    def test_先頭の句点を切り出す(self):
        assert pull_back("。こんにちは、お世話になります。") == ("。", "こんにちは、お世話になります。")

    def test_記号が無ければそのまま(self):
        assert pull_back("こんにちは") == ("", "こんにちは")

    def test_続けて並んだ記号もまとめて返す(self):
        assert pull_back("。」ところで") == ("。」", "ところで")

    def test_開き括弧は返さない(self):
        """そこから文が始まることがある。"""
        assert pull_back("（続き）です") == ("", "（続き）です")

    def test_記号だけの行(self):
        assert pull_back("。") == ("。", "")


class TestStitch:
    def test_実会議で出た形を直す(self):
        before, after = stitch("自分のマイクです", "。こんにちは、お世話になります。")

        assert before == "自分のマイクです。"
        assert after == "こんにちは、お世話になります。"

    def test_句点を二重にしない(self):
        before, after = stitch("先日はどうも。", "。今日")

        assert before == "先日はどうも。"
        assert after == "今日"

    def test_記号だけの行は空になる(self):
        """呼び手が行ごと捨てられるように。"""
        before, after = stitch("お話しありましょう", "。")

        assert before == "お話しありましょう。"
        assert after == ""

    def test_つなぐものが無ければ触らない(self):
        assert stitch("こんにちは", "お世話になります") == ("こんにちは", "お世話になります")

    def test_前の行が無くても落ちない(self):
        assert stitch("", "。ところで") == ("。", "ところで")
