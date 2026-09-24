"""ローカル UI のイベントバスを検証する。"""

from src.ui.bus import EventBus


def test_sequence_is_monotonic() -> None:
    """publish は単調増加する連番を採番する。"""
    bus = EventBus()
    assert [bus.publish("segment", {"index": index}).seq for index in range(3)] == [1, 2, 3]


def test_subscription_receives_history_before_live_events() -> None:
    """新規購読はライブ配信より先に履歴を受け取る。"""
    bus = EventBus()
    bus.publish("segment", {"index": 1})
    bus.publish("state", {"index": 2})
    subscription = bus.subscribe()
    bus.publish("segment", {"index": 3})
    assert [subscription.get(0.01).data["index"] for _ in range(3)] == [1, 2, 3]  # type: ignore[union-attr]


def test_level_is_not_replayed_from_history() -> None:
    """頻繁に流れる level は後から購読したクライアントへ再生しない。"""
    bus = EventBus()
    bus.publish("level", {"self_dbfs": -10})
    subscription = bus.subscribe()
    assert subscription.get(0.01) is None


def test_participants_is_not_replayed_and_is_available_as_latest() -> None:
    """参加者一覧は履歴に積まず、最新値だけ取得できる。"""
    bus = EventBus()
    expected = bus.publish("participants", {"items": [{"name": "田中"}]})
    subscription = bus.subscribe()
    assert subscription.get(0.01) is None
    assert bus.latest("participants") == expected


def test_relabel_is_replayed_from_history() -> None:
    """行単位の話者変更は後から接続した UI にも再生する。"""
    bus = EventBus()
    bus.publish("relabel", {"start_time": 1.0, "old": "不明話者?", "new": "田中"})
    subscription = bus.subscribe()
    event = subscription.get(0.01)
    assert event is not None
    assert event.type == "relabel"


def test_latest_returns_last_event_for_type() -> None:
    """latest は種別ごとの最後のイベントを返す。"""
    bus = EventBus()
    bus.publish("metrics", {"value": 1})
    expected = bus.publish("metrics", {"value": 2})
    assert bus.latest("metrics") == expected
    assert bus.latest("missing") is None


def test_closed_subscription_receives_nothing() -> None:
    """close 済み購読者は以後のイベントを受け取らない。"""
    bus = EventBus()
    subscription = bus.subscribe()
    subscription.close()
    bus.publish("segment", {"text": "発話"})
    assert subscription.get(0.01) is None


def test_subscription_drops_oldest_event_when_queue_is_full() -> None:
    """購読者キューが満杯なら最古のイベントを落とす。"""
    bus = EventBus()
    subscription = bus.subscribe()
    for index in range(10_001):
        bus.publish("level", {"index": index})
    first = subscription.get(0.01)
    assert first is not None
    assert first.data["index"] == 1
