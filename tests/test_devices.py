"""デバイス名前解決・レベル計測・複数出力装置の点検の単体テスト。"""

import numpy as np
import pytest

from src.audio.devices import (
    GHOST,
    OFFLINE,
    ONLINE,
    DeviceInfo,
    DeviceNotFoundError,
    LevelReading,
    MultiOutputInfo,
    SubDevice,
    find_input_devices,
    format_device_table,
    level_of,
    resolve_input_device,
)


def _devices() -> list[DeviceInfo]:
    """実測（2026-09-08）のデバイス構成を模したもの。"""
    return [
        DeviceInfo(0, "External Display", 0, 2, 48000),
        DeviceInfo(6, "Headset One", 1, 0, 48000),
        DeviceInfo(7, "Headset One", 0, 2, 48000),
        DeviceInfo(8, "BlackHole 2ch", 2, 2, 48000),
        DeviceInfo(9, "MacBook Proのマイク", 1, 0, 48000),
        DeviceInfo(13, "会議録音用（複数出力装置）", 0, 2, 48000),
    ]


class TestResolveInputDevice:
    def test_picks_the_connected_candidate(self):
        resolved = resolve_input_device(
            ["Shure MV7", "Headset One", "AirPods Pro"], devices=_devices()
        )
        assert resolved.index == 6
        assert resolved.name == "Headset One"
        assert resolved.matched_candidate == "Headset One"

    def test_candidate_order_is_priority(self):
        devices = _devices() + [DeviceInfo(3, "Shure MV7", 2, 2, 48000)]
        resolved = resolve_input_device(
            ["Shure MV7", "Headset One"], devices=devices
        )
        assert resolved.name == "Shure MV7"

    def test_never_picks_an_output_only_device(self):
        # [7] Headset One は出力専用。入力として掴んではいけない。
        resolved = resolve_input_device(["Headset One"], devices=_devices())
        assert resolved.index == 6
        assert resolved.channels == 1

    def test_matching_ignores_case_and_spaces(self):
        devices = _devices() + [DeviceInfo(20, "airpods pro3", 1, 0, 48000)]
        resolved = resolve_input_device(["AirPods Pro"], devices=devices)
        assert resolved.index == 20

    def test_partial_match(self):
        resolved = resolve_input_device(["BlackHole"], devices=_devices())
        assert resolved.name == "BlackHole 2ch"

    def test_raises_instead_of_falling_back(self):
        """見つからないとき既定デバイスを掴まない。掴むと無音事故になる。"""
        with pytest.raises(DeviceNotFoundError) as exc:
            resolve_input_device(["Shure MV7", "Yeti Nano"], devices=_devices())
        message = str(exc.value)
        assert "Shure MV7" in message          # 候補を出す
        assert "Headset One" in message    # 実在デバイスも出す

    def test_empty_candidates_raises(self):
        with pytest.raises(DeviceNotFoundError):
            resolve_input_device([], devices=_devices())

    def test_blank_candidate_is_skipped(self):
        resolved = resolve_input_device(["", "  ", "BlackHole 2ch"], devices=_devices())
        assert resolved.name == "BlackHole 2ch"


class TestInterviewDeviceResolution:
    """対面モード（Wireless Mic）の名前解決。実機で確認した実名を固定する。"""

    # Wireless Mic の USB アダプタを挿すと "Wireless microphone" 入力2/出力0 で出る
    # （実機確認 2026-09-08）。入力2ch なので L/R 物理分離の前提を満たす。
    WIRELESS_MIC = DeviceInfo(8, "Wireless microphone", 2, 0, 48000)

    def test_resolves_lark_adapter_by_name(self):
        resolved = resolve_input_device(
            ["Wireless microphone", "Wireless Mic", "WIRELESS_MIC"],
            devices=_devices() + [self.WIRELESS_MIC],
        )
        assert resolved.name == "Wireless microphone"
        assert resolved.channels == 2   # L/R 分離に2ch 要る

    def test_hardcoded_index_1_would_have_been_wrong(self):
        """旧設定の audio.device: 1 は、今は出力専用デバイスを指している。"""
        devices = _devices() + [self.WIRELESS_MIC]
        by_index = {d.index: d for d in devices}
        assert not by_index[0].is_input          # [0] BenQ = 出力専用
        assert resolve_input_device(
            ["Wireless microphone"], devices=devices
        ).index == 8                              # 名前で引けば正しく当たる


class TestFormatDeviceTable:
    def test_lists_index_and_name(self):
        table = format_device_table(_devices())
        assert "[ 8]" in table
        assert "BlackHole 2ch" in table
        assert "IN:2" in table


class TestLevelOf:
    def test_full_scale_is_zero_dbfs(self):
        reading = level_of(np.ones(4800, dtype=np.float32), 0.1)
        assert reading.peak_dbfs == pytest.approx(0.0, abs=0.01)
        assert not reading.is_silent

    def test_digital_silence(self):
        reading = level_of(np.zeros(4800, dtype=np.float32), 0.1)
        assert reading.peak_dbfs == -100.0
        assert reading.is_silent

    def test_quiet_room_tone_counts_as_silent(self):
        # -80 dBFS 相当。環境ノイズすら拾えていない水準。
        reading = level_of(np.full(4800, 1e-4, dtype=np.float32), 0.1)
        assert reading.is_silent

    def test_normal_speech_is_not_silent(self):
        reading = level_of(np.full(4800, 0.05, dtype=np.float32), 0.1)
        assert not reading.is_silent

    def test_empty_input(self):
        reading = level_of(np.array([], dtype=np.float32), 0.0)
        assert reading.is_silent

    def test_handles_stereo_input(self):
        stereo = np.zeros((4800, 2), dtype=np.float32)
        stereo[:, 1] = 1.0
        reading = level_of(stereo, 0.1)
        assert reading.peak_dbfs == pytest.approx(0.0, abs=0.01)


class TestFindInputDevices:
    def test_returns_every_connected_candidate_in_priority_order(self):
        """複数の候補機を同時に挿している日がある（実際に4台同時が起きた）。"""
        devices = _devices() + [
            DeviceInfo(3, "Shure MV7", 2, 2, 48000),
            DeviceInfo(20, "Bose QC Earbuds", 1, 0, 48000),
        ]
        found = find_input_devices(
            ["Headset One", "Shure MV7", "AirPods Pro", "Bose QC Earbuds"],
            devices,
        )
        assert [d.name for d in found] == ["Headset One", "Shure MV7", "Bose QC Earbuds"]

    def test_no_duplicates_when_candidates_overlap(self):
        found = find_input_devices(["Headset One", "Headset One"], _devices())
        assert len(found) == 1

    def test_empty_when_nothing_connected(self):
        assert find_input_devices(["Yeti Nano"], _devices()) == []


class TestMultiOutputInfo:
    def _info(self) -> MultiOutputInfo:
        """実測（2026-09-08 夕・自分さんが機器を一通り接続した状態）の構成。"""
        return MultiOutputInfo(
            name="会議録音用（複数出力装置）",
            uid="~:AMS2_StackedOutput:0",
            subdevices=[
                SubDevice("BlackHole2ch_UID", "BlackHole 2ch", False),
                SubDevice("00-00-5E-00-53-FF:output", "", False),
                SubDevice("00-00-5E-00-53-A1:output", "airpods pro3", False),
                SubDevice("AppleUSBAudioEngine:Shure Inc:Shure MV7:MV7i#7:2,3", "Shure MV7", False),
                SubDevice("SoundIDReference_DeviceUID", "", False),
                SubDevice("AppleUSBAudioEngine:Headset:Headset One:24DB:4", "Headset One", False),
                SubDevice("00-00-5E-00-53-B2:output", "Bose QC Earbuds", False),
                SubDevice("AppleUSBAudioEngine:C-Media:USB Audio Device:143000:2,1", "USB Audio Device", True),
            ],
        )

    # ペアリング済み BT。airpods pro3 と Bose は居るが 00-00-5E-00-53-FF は居ない。
    # MAC は RFC 7042 のドキュメント用ブロック（00-00-5E-00-53-xx）。実機の値ではない。
    PAIRED = {"00:00:5E:00:53:A1", "00:00:5E:00:53:B2", "00:00:5E:00:53:C3"}

    def test_contains_registered_device(self):
        assert self._info().contains("BlackHole 2ch")
        assert self._info().contains("Headset One")

    def test_contains_ignores_case_and_spaces(self):
        assert self._info().contains("AirPods Pro")

    def test_detects_missing_device(self):
        assert not self._info().contains("MacBook Proのスピーカー")

    def test_unpaired_bluetooth_entry_is_a_ghost(self):
        """ペアリング一覧に無い BT 機器＝二度と戻らない残骸。外してよい。"""
        ghosts = self._info().ghosts(self.PAIRED)
        assert [g.uid for g in ghosts] == ["00-00-5E-00-53-FF:output"]

    def test_offline_virtual_driver_is_not_a_ghost(self):
        """SoundID Reference はアプリ未起動でオフラインになるだけ。外してはいけない。

        「名前が空＝幽霊」で判定すると、これを誤って残骸と呼んでしまう。
        """
        info = self._info()
        soundid = info.subdevices[4]
        assert soundid.is_offline
        assert info.status(soundid, self.PAIRED) == OFFLINE
        assert soundid not in info.ghosts(self.PAIRED)

    def test_status_of_present_device(self):
        info = self._info()
        assert info.status(info.subdevices[0], self.PAIRED) == ONLINE

    def test_ghost_classification_is_suspended_without_pairing_list(self):
        """BT 一覧が取れないときは GHOST 断定を避ける（誤って外させない）。"""
        info = self._info()
        assert info.status(info.subdevices[1], None) == OFFLINE
        assert info.ghosts(None) == []

    def test_bluetooth_mac_is_extracted(self):
        assert SubDevice("00-00-5E-00-53-B2:output").bluetooth_mac == "00:00:5E:00:53:B2"

    def test_non_bluetooth_uid_has_no_mac(self):
        assert SubDevice("BlackHole2ch_UID").bluetooth_mac == ""

    def test_online_excludes_offline_entries(self):
        assert len(self._info().online()) == 6

    def test_audible_outputs_excludes_the_recording_tap(self):
        """BlackHole は音を鳴らさない。ここが空だと Zoom の音がどこにも聞こえない。"""
        audible = self._info().audible_outputs(["BlackHole 2ch"])
        assert [s.name for s in audible] == [
            "airpods pro3", "Shure MV7", "Headset One",
            "Bose QC Earbuds", "USB Audio Device",
        ]

    def test_audible_outputs_empty_means_nothing_can_be_heard(self):
        info = MultiOutputInfo(
            name="x", uid="y",
            subdevices=[
                SubDevice("BlackHole2ch_UID", "BlackHole 2ch"),
                SubDevice("SoundIDReference_DeviceUID", ""),   # オフライン
            ],
        )
        assert info.audible_outputs(["BlackHole 2ch"]) == []

    def test_mic_in_bundle_is_not_the_right_question(self):
        """据え置きマイクは「聴く側」と別デバイスでよい。

        Shure MV7 で喋り USB Audio Device で聴く運用があるので、
        「今日のマイクが束に入っているか」では可否を判定できない。
        """
        info = MultiOutputInfo(
            name="x", uid="y",
            subdevices=[
                SubDevice("BlackHole2ch_UID", "BlackHole 2ch"),
                SubDevice("usb", "USB Audio Device", True),
            ],
        )
        assert not info.contains("Shure MV7")      # マイクは束に居ない
        assert info.audible_outputs(["BlackHole 2ch"])  # それでも聴こえる＝問題なし

    def test_drift_correction_is_parsed(self):
        by_name = {s.name: s for s in self._info().subdevices}
        assert by_name["USB Audio Device"].drift_correction
        assert not by_name["airpods pro3"].drift_correction
