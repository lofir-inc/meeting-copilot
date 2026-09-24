"""この会議はどのクライアントの会議か — 議事録・タスク・辞書の行き先。"""

from __future__ import annotations

import json

import pytest

from src import clients

MASTER = [
    {"name": "自社", "client_id": "0000", "minutes_db_id": "db-your-org"},
    {"name": "株式会社ミナト", "client_id": "0045", "minutes_db_id": "db-minato"},
    {"name": "sample株式会社", "client_id": "sample", "minutes_db_id": ""},
    {"name": "取引先C株式会社", "client_id": "0054", "minutes_db_id": "db-techno"},
]


def test_自社を先頭にテストを最後に並べる():
    picked = clients.options(MASTER)

    assert [entry["name"] for entry in picked] == [
        "自社", "取引先C株式会社", "株式会社ミナト", "sample株式会社"]
    assert picked[-1]["test"] is True and picked[0]["test"] is False
    assert picked[-1]["has_minutes_db"] is False


def test_選ばなければ自社へ倒れる(tmp_path):
    assert clients.effective(tmp_path) == ("自社", False)


def test_選べば行き先が変わる(tmp_path):
    clients.choose(tmp_path, "株式会社ミナト", client_id="0045")

    assert clients.effective(tmp_path) == ("株式会社ミナト", True)
    assert json.loads((tmp_path / clients.CLIENT_FILE).read_text(encoding="utf-8"))["client_id"] == "0045"


def test_空の名前は受け付けない(tmp_path):
    """黙って自社に倒さない（選んだのか選んでいないのかが分からなくなる）。"""
    with pytest.raises(ValueError):
        clients.choose(tmp_path, "  ")


def test_控えを書いて読む(tmp_path):
    clients.save_cache(tmp_path / "clients.json", MASTER)

    assert len(clients.load_cache(tmp_path / "clients.json")) == 4
    assert clients.load_cache(tmp_path / "無い.json") == []


def test_一覧はタスク管理の共有モジュールから読む(tmp_path):
    shared = tmp_path / "_shared"
    shared.mkdir()
    (shared / "task_hub_context.py").write_text("CLIENT_MASTER_DB = 'db'\n", encoding="utf-8")
    (shared / "task_hub_notion.py").write_text(
        "def query_database(db_id):\n"
        "    return [{'properties': {}}]\n"
        "def prop_title(props, name):\n"
        "    return '株式会社ミナト'\n"
        "def prop_text(props, name):\n"
        "    return '0045' if name == 'client_id' else 'db-minato'\n", encoding="utf-8")

    assert clients.fetch(shared) == [
        {"name": "株式会社ミナト", "client_id": "0045", "minutes_db_id": "db-minato", "people": []}]


def test_その会社の人も控える(tmp_path):
    """People Master の人（名前＋話者ラベル）を同期のときに控える。

    議事録を作るときに Notion を叩かないため（仕上げの経路に失敗する箇所を増やさない）。
    """
    shared = tmp_path / "_shared"
    shared.mkdir()
    (shared / "task_hub_context.py").write_text("CLIENT_MASTER_DB = 'db'\n", encoding="utf-8")
    (shared / "task_hub_notion.py").write_text(
        "def query_database(db_id):\n"
        "    if db_id == 'people':\n"
        "        return [{'id': 'p1', 'properties': {'Organization': {'relation': [{'id': 'c1'}]}}}]\n"
        "    return [{'id': 'c1', 'properties': {}}]\n"
        "def prop_title(props, name=None):\n"
        "    return '株式会社ミナト' if name else '南社長'\n"
        "def prop_text(props, name):\n"
        "    return '0045' if name == 'client_id' else 'db-minato'\n"
        "def prop_multi_select(props, name):\n"
        "    return ['MINATO南社長']\n", encoding="utf-8")

    found = clients.fetch(shared, people_db="people")

    assert found[0]["people"] == ["南社長", "MINATO南社長"]
    assert clients.people_of(found, "株式会社ミナト") == ["南社長", "MINATO南社長"]


def test_人が読めなくても一覧は返す(tmp_path):
    """People Master が読めないだけで議事録が作れなくなるのは困る（fail-open）。"""
    shared = tmp_path / "_shared"
    shared.mkdir()
    (shared / "task_hub_context.py").write_text("CLIENT_MASTER_DB = 'db'\n", encoding="utf-8")
    (shared / "task_hub_notion.py").write_text(
        "def query_database(db_id):\n"
        "    if db_id == 'people':\n"
        "        raise RuntimeError('People Master に届きません')\n"
        "    return [{'id': 'c1', 'properties': {}}]\n"
        "def prop_title(props, name=None):\n"
        "    return '株式会社ミナト'\n"
        "def prop_text(props, name):\n"
        "    return '0045' if name == 'client_id' else 'db-minato'\n"
        "def prop_multi_select(props, name):\n"
        "    return []\n", encoding="utf-8")

    found = clients.fetch(shared, people_db="people")

    assert found[0]["name"] == "株式会社ミナト" and found[0]["people"] == []


def test_共有モジュールが無ければ例外(tmp_path):
    with pytest.raises(FileNotFoundError):
        clients.fetch(tmp_path)
