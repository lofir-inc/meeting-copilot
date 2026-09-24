"""相づち判定の単体テスト。

例は 2026-09-11 の実会議 50 分（945 行）から拾ったものを使う。
判定は保守的に — 中身のある行に印を付けてしまうと、議事録から消える側に回る。
"""

import pytest

from src.text.aizuchi import is_aizuchi, mark_rows


@pytest.mark.parametrize("text", [
    "うん。", "はい。", "うんうん。", "うんうんうん。", "ああ。",
    "はいはいはい。", "そう。", "うん、うん。", "うん。はい。", "ええ。",
    "えっと", "あの", "なるほどなるほど。はい。", "うんうん。うんうんうん。なるほどなるほど。",
    "うん,うん,うん。はい。",   # 半角カンマ混じり（NFKC で正規化してから判定する）
])
def test_相づちだけの行には印が付く(text):
    assert is_aizuchi(text)


@pytest.mark.parametrize("text", [
    "了解です。", "そうです。", "お願いします。", "すげえ。", "すごいねこれ。",
    "はい。ぜひぜひ。", "うん、確かに。", "そうですね。", "あ、いいよ。",
    "提案書", "空間提案プラン", "あと新商品", "叩き台ですね。",
])
def test_中身のある行には付かない(text):
    assert not is_aizuchi(text)


def test_空行は相づちではない():
    """「何も無い」は相づちとは別物。落とし方が変わるので混ぜない。"""
    assert not is_aizuchi("")
    assert not is_aizuchi("。。。")


def test_mark_rows_は行を消さずに印だけ付ける():
    rows = [{"text": "うん。"}, {"text": "叩き台ですね。"}, {"text": "はいはい。"}]
    marked = mark_rows(rows)
    assert marked == 2
    assert len(rows) == 3
    assert [row["aizuchi"] for row in rows] == [True, False, True]


def test_mark_rows_は無効にできる():
    rows = [{"text": "うん。"}]
    assert mark_rows(rows, enabled=False) == 0
    assert "aizuchi" not in rows[0]
