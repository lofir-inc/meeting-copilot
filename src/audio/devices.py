"""オーディオデバイスの名前解決・レベル計測・出力系統の点検。

番号直書きをやめるための土台。
接続機器で sounddevice のインデックスは毎回変わるため、設定には「候補名のリスト」を
持ち、起動時に「今つながっている物」を選ぶ。見つからなければ黙って別デバイスを掴まず、
候補と実在デバイスの一覧を添えて停止する（無音で録れていない事故を防ぐ）。
"""

from __future__ import annotations

import logging
import plistlib
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# macOS の CoreAudio 設定。複数出力装置（stacked output）の構成がここに入っている。
_SYSTEM_AUDIO_PLIST = "/Library/Preferences/Audio/com.apple.audio.SystemSettings.plist"


class DeviceNotFoundError(RuntimeError):
    """候補名のどれにも一致する入力デバイスが無い。"""


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


@dataclass
class ResolvedDevice:
    """候補名リストから解決された1デバイス。"""
    index: int
    name: str
    channels: int
    matched_candidate: str

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} (ch={self.channels})"


@dataclass
class LevelReading:
    """入力レベルの実測値。"""
    peak_dbfs: float
    rms_dbfs: float
    seconds: float

    # これを下回ったら「無音」とみなす（環境ノイズすら拾えていない水準）
    SILENT_DBFS = -70.0

    @property
    def is_silent(self) -> bool:
        return self.peak_dbfs < self.SILENT_DBFS

    @property
    def is_empty(self) -> bool:
        """音声が1フレームも届いていない（デバイスを掴めていない）。

        「無音」とは原因が違う。無音はミュートや発言なし、こちらは機器・許可の問題。
        2026-09-12 の実機で、マイクと相手側の両方が -100 dBFS（＝空）になった。
        """
        return self.peak_dbfs <= -100.0

    def __str__(self) -> str:
        return f"peak={self.peak_dbfs:6.1f} dBFS / rms={self.rms_dbfs:6.1f} dBFS"


# Bluetooth 機器の UID は "00-00-5E-00-53-B2:output" の形（MAC アドレス）。
_BT_UID_RE = re.compile(r"^((?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}):(?:output|input)$")

# サブデバイスの状態
ONLINE = "online"      # 今この Mac に見えている
OFFLINE = "offline"    # 今は見えないが、繋げば戻る（アプリ未起動の仮想ドライバ・未接続のUSB等）
GHOST = "ghost"        # 二度と戻らない残骸（ペアリングを解除した Bluetooth 機器）


@dataclass
class SubDevice:
    """複数出力装置に束ねられた1サブデバイス。"""
    uid: str
    name: str = ""
    drift_correction: bool = False

    @property
    def is_offline(self) -> bool:
        """束に入っているが、今この Mac に見えていないか。

        macOS は解決できなかったサブデバイスを Audio MIDI 設定で「オフライン装置」と
        表示し、plist には名前を書かない。

        **これだけでは「もう戻らない残骸」と「今日たまたま繋いでいないだけ」を
        区別できない。** 判定は `MultiOutputInfo.status()` で行うこと。
        SoundID Reference のような仮想ドライバは、アプリを起動していないだけで
        オフラインになる（＝外してはいけない）。
        """
        return not self.name

    @property
    def bluetooth_mac(self) -> str:
        """Bluetooth 機器の UID なら MAC アドレスを返す。それ以外は空文字。"""
        match = _BT_UID_RE.match(self.uid)
        return match.group(1).replace("-", ":").upper() if match else ""


@dataclass
class MultiOutputInfo:
    """複数出力装置（stacked output）の構成。"""
    name: str
    uid: str
    subdevices: list[SubDevice] = field(default_factory=list)

    def contains(self, device_name: str) -> bool:
        """指定名のデバイスが束に入っているか（部分一致・大小無視）。"""
        needle = _normalize(device_name)
        return any(needle in _normalize(s.name) for s in self.subdevices if s.name)

    def status(self, sub: SubDevice, paired_macs: set[str] | None = None) -> str:
        """サブデバイス1件の状態を ONLINE / OFFLINE / GHOST で返す。

        オフラインの Bluetooth 機器のうち、**ペアリング一覧に無いものだけ**を GHOST とする。
        ペアリングを解除した機器は二度と戻らないため。逆に SoundID Reference のような
        仮想ドライバはアプリ未起動でオフラインになるだけなので、外してはいけない。
        """
        if not sub.is_offline:
            return ONLINE

        mac = sub.bluetooth_mac
        if mac and paired_macs is not None and mac not in paired_macs:
            return GHOST

        return OFFLINE

    def classified(self, paired_macs: set[str] | None = None) -> list[tuple[SubDevice, str]]:
        """全サブデバイスを (装置, 状態) の並びで返す（束の順序を保つ）。"""
        return [(s, self.status(s, paired_macs)) for s in self.subdevices]

    def ghosts(self, paired_macs: set[str] | None = None) -> list[SubDevice]:
        """二度と戻らない残骸だけを返す。"""
        return [s for s, state in self.classified(paired_macs) if state == GHOST]

    def online(self) -> list[SubDevice]:
        """今この Mac に見えているサブデバイスだけを返す。"""
        return [s for s in self.subdevices if not s.is_offline]

    def audible_outputs(self, tap_names: list[str] | None = None) -> list[SubDevice]:
        """「自分の耳に届く」経路になりうるサブデバイスを返す。

        録音タップ（BlackHole）は音を鳴らさないので除く。ここが空だと、Zoom の音が
        **どこにも聞こえない**構成になっている。マイクと聴くデバイスは別でよい
        （例: Shure MV7 で喋り、USB Audio Device の有線イヤホンで聴く）ので、
        「今日のマイクが束に入っているか」で代用してはいけない。
        """
        taps = [_normalize(n) for n in (tap_names or ["BlackHole"])]
        return [
            s for s in self.online()
            if not any(tap in _normalize(s.name) for tap in taps)
        ]


def _normalize(text: str) -> str:
    """比較用の正規化 — 小文字化して空白を落とす。"""
    return "".join(text.lower().split())


def list_devices() -> list[DeviceInfo]:
    """現在見えているオーディオデバイスを全件返す。"""
    devices = []
    for index, raw in enumerate(sd.query_devices()):
        devices.append(
            DeviceInfo(
                index=index,
                name=raw["name"],
                max_input_channels=raw["max_input_channels"],
                max_output_channels=raw["max_output_channels"],
                default_samplerate=raw["default_samplerate"],
            )
        )
    return devices


def format_device_table(devices: list[DeviceInfo] | None = None) -> str:
    """デバイス一覧を人が読める表にする（エラーメッセージに添える用）。"""
    if devices is None:
        devices = list_devices()
    lines = []
    for d in devices:
        tags = []
        if d.max_input_channels:
            tags.append(f"IN:{d.max_input_channels}")
        if d.max_output_channels:
            tags.append(f"OUT:{d.max_output_channels}")
        lines.append(f"  [{d.index:2d}] {' '.join(tags):12s} {d.name}")
    return "\n".join(lines)


def find_input_devices(
    candidates: list[str],
    devices: list[DeviceInfo] | None = None,
) -> list[ResolvedDevice]:
    """候補名に一致する入力デバイスを**優先順にすべて**返す。

    複数の候補機を同時に挿していることがある（自分さんは Headset One・MV7i・AirPods・
    Bose を同時接続する日がある）。1つに決め打つ前に「他にも候補がある」ことを
    呼び出し側へ見せられるようにしておく。
    """
    if devices is None:
        devices = list_devices()
    inputs = [d for d in devices if d.is_input]

    found: list[ResolvedDevice] = []
    seen: set[int] = set()

    for candidate in candidates:
        needle = _normalize(candidate)
        if not needle:
            continue
        for device in inputs:
            if needle in _normalize(device.name) and device.index not in seen:
                seen.add(device.index)
                found.append(
                    ResolvedDevice(
                        index=device.index,
                        name=device.name,
                        channels=device.max_input_channels,
                        matched_candidate=candidate,
                    )
                )
    return found


def resolve_input_device(
    candidates: list[str],
    *,
    label: str = "入力デバイス",
    devices: list[DeviceInfo] | None = None,
) -> ResolvedDevice:
    """候補名のリストから、今つながっている入力デバイスを1つ選ぶ。

    候補は**優先順**に並べる。前方の候補ほど優先される。
    照合は部分一致・大小無視・空白無視（"AirPods Pro" と "airpods pro3" が一致する）。

    見つからなければ DeviceNotFoundError を送出する。**既定デバイスへのフォールバックはしない。**
    黙って別のデバイスを掴むと「録れているつもりで無音」になるため。
    """
    if not candidates:
        raise DeviceNotFoundError(f"{label}: 候補名が1つも設定されていません。")

    if devices is None:
        devices = list_devices()

    found = find_input_devices(candidates, devices)
    if found:
        resolved = found[0]
        logger.info("%s を解決: %s ← 候補 %r", label, resolved, resolved.matched_candidate)
        if len(found) > 1:
            logger.info(
                "%s の他の候補も接続中: %s",
                label,
                "、".join(str(d) for d in found[1:]),
            )
        return resolved

    inputs = [d for d in devices if d.is_input]
    raise DeviceNotFoundError(
        f"{label}: 候補名のどれにも一致する入力デバイスがありません。\n"
        f"  候補: {candidates}\n"
        f"接続中の入力デバイス:\n{format_device_table(inputs)}\n"
        f"→ 機器をつなぐか、config/settings.yaml の候補名を実名に合わせてください。"
    )


def paired_bluetooth_macs() -> set[str] | None:
    """ペアリング済み Bluetooth 機器の MAC アドレス一覧を返す。

    これが「二度と戻らない残骸」と「今日つないでいないだけ」を分ける物差しになる。
    取得できなければ None（判定を保留し、GHOST 断定を避ける）。
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPBluetoothDataType"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout:
            return None
    except Exception:
        logger.debug("Bluetooth 一覧の取得に失敗", exc_info=True)
        return None

    macs = set()
    for match in re.finditer(r"Address:\s*([0-9A-Fa-f:%-]{17})", result.stdout):
        macs.add(match.group(1).replace("-", ":").upper())
    return macs or None


def measure_level(
    device: int,
    *,
    channels: int = 1,
    seconds: float = 2.0,
    sample_rate: int = 48000,
) -> LevelReading:
    """指定デバイスを数秒モニタして入力レベル（dBFS）を実測する。"""
    frames = int(seconds * sample_rate)
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return level_of(audio, seconds)


def level_of(audio: np.ndarray, seconds: float) -> LevelReading:
    """録音済み配列からピーク／RMS の dBFS を出す。"""
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return LevelReading(peak_dbfs=-100.0, rms_dbfs=-100.0, seconds=seconds)
    peak = float(np.max(np.abs(flat)))
    rms = float(np.sqrt(np.mean(flat**2)))
    return LevelReading(
        peak_dbfs=_to_dbfs(peak),
        rms_dbfs=_to_dbfs(rms),
        seconds=seconds,
    )


def _to_dbfs(amplitude: float) -> float:
    if amplitude <= 0:
        return -100.0
    return float(20 * np.log10(amplitude))


def default_output_name() -> str:
    """現在の既定出力デバイス名を返す（取れなければ空文字）。"""
    try:
        _, out_index = sd.default.device
        if out_index is None or out_index < 0:
            return ""
        return sd.query_devices(out_index)["name"]
    except Exception:
        logger.debug("既定出力デバイスの取得に失敗", exc_info=True)
        return ""


def read_multi_output_info(name_hint: str) -> MultiOutputInfo | None:
    """複数出力装置（stacked output）の構成を CoreAudio の設定から読む。

    Audio MIDI 設定を開かなくても中身が分かる。`system_profiler` はこの構成を出さない。
    読めなければ None（macOS 以外・plist が無い・権限が無い等）。
    """
    try:
        raw = subprocess.run(
            ["plutil", "-convert", "xml1", "-o", "-", _SYSTEM_AUDIO_PLIST],
            capture_output=True,
            timeout=10,
        )
        if raw.returncode != 0 or not raw.stdout:
            return None
        settings = plistlib.loads(raw.stdout)
    except Exception:
        logger.debug("CoreAudio 設定の読み取りに失敗", exc_info=True)
        return None

    needle = _normalize(name_hint)
    for key, value in settings.items():
        if not key.startswith("MetaDevice.") or not isinstance(value, dict):
            continue
        if not value.get("stacked"):
            continue
        device_name = str(value.get("name", ""))
        if needle and needle not in _normalize(device_name):
            continue

        subdevices = []
        for entry in value.get("subdevices", []):
            if not isinstance(entry, dict):
                continue
            subdevices.append(
                SubDevice(
                    uid=str(entry.get("uid", "")),
                    name=str(entry.get("name", "")),
                    drift_correction=bool(int(entry.get("drift", 0) or 0)),
                )
            )
        return MultiOutputInfo(
            name=device_name,
            uid=str(value.get("uid", "")),
            subdevices=subdevices,
        )

    return None
