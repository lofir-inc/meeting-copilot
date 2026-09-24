"""設定を画面から直す — 話者名と Notion のペルソナ名の対応表。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src import settings_edit

SETTINGS = """# 会議システムの設定
task_hub:
  client_name: 自社            # 自社
  personas:                     # 会議の話者名 → Notion のペルソナ名
                                # この注記は消えてはいけない
    自分: 自社社長
  notion_tasks: true            # この行も残る

meeting:
  self_name: 自分
"""

WITHOUT = """task_hub:
  client_name: 自社
  notion_tasks: true

meeting:
  self_name: 自分
"""


def write(tmp_path: Path, text: str = SETTINGS) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_いまの対応表を読む(tmp_path):
    assert settings_edit.read_personas(write(tmp_path)) == {"自分": ["自社社長"]}


def test_足しても注記が消えない(tmp_path):
    """設定はコメントが中身の半分（yaml.dump で書き直さない）。"""
    path = write(tmp_path)

    table = settings_edit.set_persona(path, "山田", "さくら歯科 歯科衛生士K")

    text = path.read_text(encoding="utf-8")
    assert table == {"自分": ["自社社長"], "山田": ["さくら歯科 歯科衛生士K"]}
    assert "この注記は消えてはいけない" in text and "この行も残る" in text
    assert yaml.safe_load(text)["meeting"]["self_name"] == "自分"


def test_同じ人を直すと置き換わる(tmp_path):
    path = write(tmp_path)

    settings_edit.set_persona(path, "自分", "自社開発者")

    assert settings_edit.read_personas(path) == {"自分": ["自社開発者"]}
    assert path.read_text(encoding="utf-8").count("自分:") == 1


def test_空にすると外れる(tmp_path):
    path = write(tmp_path)

    assert settings_edit.set_persona(path, "自分", "") == {}
    assert "personas:" in path.read_text(encoding="utf-8")       # 見出しは残す


def test_節が無ければ作る(tmp_path):
    path = write(tmp_path, WITHOUT)

    table = settings_edit.set_persona(path, "自分", "自社社長")

    assert table == {"自分": ["自社社長"]}
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["task_hub"]["notion_tasks"] is True


def test_記号を含む名前も壊さない(tmp_path):
    path = write(tmp_path)

    settings_edit.set_persona(path, "参加者A", "さくら歯科: 院長")

    assert settings_edit.read_personas(path)["参加者A"] == ["さくら歯科: 院長"]


def test_直す前の設定を退避する(tmp_path):
    """直接上書きしない（戻せるようにする）。"""
    path = write(tmp_path)
    trash = tmp_path / "99-trash" / "settings"

    settings_edit.set_persona(path, "山田", "誰か", trash_dir=trash)

    saved = list(trash.glob("*settings.yaml"))
    assert len(saved) == 1 and "自社社長" in saved[0].read_text(encoding="utf-8")


def test_名前が空なら断る(tmp_path):
    with pytest.raises(ValueError):
        settings_edit.set_persona(write(tmp_path), "  ", "自社社長")


def test_1人に複数のペルソナを紐づける(tmp_path):
    """連携先の設定に合わせる（運用者 指摘 2026-09-17）。"""
    path = write(tmp_path)

    table = settings_edit.set_persona(path, "自分", "自社社長、自社開発者")

    assert table == {"自分": ["自社社長", "自社開発者"]}
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["task_hub"]["personas"]["自分"] == [
        "自社社長", "自社開発者"]


def test_区切りは読点でもカンマでもよい(tmp_path):
    assert settings_edit.split_names("自社社長, 自社開発者") == ["自社社長", "自社開発者"]
    assert settings_edit.split_names(["A", " B "]) == ["A", "B"]


FULL = """meeting:
  self_name: 自分               # 自分の発話に付ける固定ラベル
  open_folder_after: true       # 終了後にフォルダを開く
  finish_mode: ask              # ask | auto | later
  screen_capture:
    enabled: false              # 既定 off
    interval_sec: 6             # 何秒ごとに見るか
    ocr: true
  voice_library:
    enabled: true               # 声紋は個人を識別できる情報
  external_stt:
    enabled: false              # 既定 off

minutes:
  engine: auto

task_hub:
  slack: true
  notion_tasks: true
  dictionary_sync: true
"""


class TestFields:
    """画面から直せる設定（機能の ON/OFF・よく触る値）。"""

    def test_設定にある項目だけを値つきで出す(self, tmp_path):
        path = write(tmp_path, FULL)

        fields = {one["key"]: one for one in settings_edit.read_fields(path)}

        assert fields["meeting.screen_capture.enabled"]["value"] is False
        assert fields["meeting.screen_capture.enabled"]["kind"] == "bool"
        assert fields["meeting.external_stt.enabled"]["danger"] is True      # 外へ出る設定は印を付ける
        assert "meeting.silence_alert_sec" not in fields                     # 設定に無い項目は出さない

    def test_入れ子の値を切り替えてもコメントが残る(self, tmp_path):
        path = write(tmp_path, FULL)

        assert settings_edit.set_value(path, "meeting.screen_capture.enabled", True) is True

        text = path.read_text(encoding="utf-8")
        assert "enabled: true" in text and "既定 off" in text
        assert yaml.safe_load(text)["meeting"]["voice_library"]["enabled"] is True   # 同名キーを巻き込まない

    def test_同じ名前のキーを取り違えない(self, tmp_path):
        """enabled は 3 か所にある（画面共有・声の台帳・外の文字起こし）。"""
        path = write(tmp_path, FULL)

        settings_edit.set_value(path, "meeting.external_stt.enabled", True)

        data = yaml.safe_load(path.read_text(encoding="utf-8"))["meeting"]
        assert data["external_stt"]["enabled"] is True
        assert data["screen_capture"]["enabled"] is False and data["voice_library"]["enabled"] is True

    def test_選べない値は断る(self, tmp_path):
        path = write(tmp_path, FULL)

        with pytest.raises(ValueError):
            settings_edit.set_value(path, "meeting.finish_mode", "すぐ")

    def test_画面に出していない設定は直せない(self, tmp_path):
        """全部を画面から触れるようにはしない（触ってはいけない値がある）。"""
        path = write(tmp_path, FULL)

        with pytest.raises(ValueError, match="画面から直せる"):
            settings_edit.set_value(path, "task_hub.chat_notify_url", "https://example.invalid/")

    def test_数字も入れられる(self, tmp_path):
        path = write(tmp_path, FULL)

        assert settings_edit.set_value(path, "meeting.screen_capture.interval_sec", 10) == 10
