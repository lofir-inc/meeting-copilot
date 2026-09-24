#!/usr/bin/env python3
"""会議モードのオーディオ経路を点検する。

会議の前に1回叩く。`会議録音用（複数出力装置）` の構成と、今日のマイクが
解決できるかを機械で確かめる。Audio MIDI 設定を直したあとの答え合わせにも使う。

    python scripts/check_audio_routing.py
    python scripts/check_audio_routing.py --config config/settings.yaml

終了コード: 0 = 問題なし / 1 = 要対処
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.audio import devices as dev  # noqa: E402

OK = "✅"
NG = "❌"
WARN = "⚠️ "


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"  {OK} {message}")

    def fail(self, message: str, fix: str = "") -> None:
        print(f"  {NG} {message}")
        if fix:
            print(f"       → {fix}")
        self.failures.append(message)

    def warn(self, message: str, fix: str = "") -> None:
        print(f"  {WARN}{message}")
        if fix:
            print(f"       → {fix}")
        self.warnings.append(message)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * 66)


def check_devices(meeting: dict, report: Report) -> tuple[object | None, object | None]:
    section("【1】入力デバイスの解決")

    mic = loopback = None
    try:
        mic = dev.resolve_input_device(meeting.get("mic_candidates", []), label="自分のマイク")
        report.ok(f"自分のマイク : {mic}  ← 候補 {mic.matched_candidate!r}")
    except dev.DeviceNotFoundError as exc:
        report.fail("自分のマイクを解決できません", str(exc).splitlines()[0])

    try:
        loopback = dev.resolve_input_device(
            meeting.get("loopback_candidates", ["BlackHole 2ch"]), label="ループバック"
        )
        report.ok(f"ループバック : {loopback}")
    except dev.DeviceNotFoundError:
        report.fail(
            "ループバックデバイスがありません",
            "BlackHole 2ch がインストールされているか確認してください。",
        )

    return mic, loopback


def check_multi_output(meeting: dict, mic, loopback, report: Report) -> None:
    target = meeting.get("multi_output_name", "会議録音用（複数出力装置）")
    section(f"【2】複数出力装置『{target}』の構成")

    info = dev.read_multi_output_info(target)
    if info is None:
        report.fail(
            f"『{target}』の構成を読めません",
            "Audio MIDI 設定で複数出力装置が作られているか確認してください。",
        )
        return

    paired = dev.paired_bluetooth_macs()
    marks = {dev.GHOST: NG, dev.OFFLINE: "･ ", dev.ONLINE: "  "}

    print(f"  {info.name}")
    for i, (sub, state) in enumerate(info.classified(paired), 1):
        drift = "音ずれ補正 ON " if sub.drift_correction else "音ずれ補正 OFF"
        role = "（プライマリ）" if i == 1 else ""
        label = sub.name or "(オフライン装置)"
        note = ""
        if state == dev.GHOST:
            note = "  ← 二度と戻らない残骸。外すこと"
        elif state == dev.OFFLINE:
            note = "  ← 今つないでいないだけ。外さないこと"
        print(f"    {marks[state]}{i}. {drift}  {label}{role}{note}")
    print()

    # --- 幽霊エントリ（ペアリング一覧にも無い＝二度と戻らない） ---
    ghosts = info.ghosts(paired)
    if paired is None:
        report.warn("Bluetooth 一覧を取得できず、残骸の判定を保留しました")
    elif ghosts:
        report.fail(
            f"二度と戻らないサブデバイスが {len(ghosts)} 件残っています"
            f"（{', '.join(g.uid for g in ghosts)}）",
            "Audio MIDI 設定で『使用』のチェックを外してください。",
        )
    else:
        report.ok("残骸サブデバイスなし")

    offline = [s for s, st in info.classified(paired) if st == dev.OFFLINE]
    if offline:
        report.ok(
            f"オフライン装置 {len(offline)} 件は正当（外さないこと）: "
            f"{', '.join(s.uid for s in offline)}"
        )

    # --- 録音タップ ---
    if loopback is not None:
        if info.contains(loopback.name):
            report.ok(f"{loopback.name} が束に入っている（録音タップ）")
        else:
            report.fail(
                f"{loopback.name} が束に入っていません",
                "この構成では録音できません。Audio MIDI 設定で追加してください。",
            )

    # --- 自分の耳に届く経路があるか ---
    # マイクと聴くデバイスは別でよい（MV7i で喋り、有線イヤホンで聴く）ので、
    #   「今日のマイクが束に入っているか」では判定できない。
    tap = [loopback.name] if loopback is not None else None
    audible = info.audible_outputs(tap)
    if audible:
        report.ok(f"聴こえる経路がある: {'、'.join(s.name for s in audible)}")
    else:
        report.fail(
            "今つながっている再生デバイスが束に1台もありません",
            "このままだと会議アプリの音がどこにも聞こえません。今日聴くデバイスを追加してください。",
        )

    # --- 候補マイクの網羅（その機器を使う日に「聴けるか」） ---
    # 束に入っている必要があるのは聴く側であって、マイクではない。
    #   据え置きマイク（MV7i）は有線イヤホンで聴くので、MV7i 自体が束に無くても構わない。
    listening_for = meeting.get("listening_device_for", {}) or {}
    missing = []
    for mic_name in meeting.get("mic_candidates", []):
        listening = listening_for.get(mic_name, mic_name)
        if not info.contains(listening):
            missing.append(f"{mic_name} → 聴く側『{listening}』")

    if missing:
        report.warn(
            "その機器を使う日に音が聞こえない組み合わせがあります: " + "、".join(missing),
            "Audio MIDI 設定で、聴く側のデバイスを束に追加してください。",
        )
    else:
        report.ok("どの候補マイクの日でも聴く経路がある（日替わりでも困らない）")

    # 今日のマイクに対応する聴く側が束に入っているか（最重要）
    if mic is not None:
        listening = listening_for.get(mic.name, mic.name)
        if info.contains(listening):
            report.ok(f"今日は『{mic.name}』で喋り『{listening}』で聴く — どちらも整合")
        else:
            report.fail(
                f"今日のマイク『{mic.name}』に対応する聴く側『{listening}』が束にありません",
                "このままだと会議アプリの音が自分に聞こえません。",
            )

    # --- 音ずれ補正 ---
    online = info.online()
    no_drift = [s for s in online if not s.drift_correction]
    if len(no_drift) == 1:
        report.ok(f"音ずれ補正 OFF はプライマリ1台のみ（{no_drift[0].name}）")
    elif len(no_drift) == 0:
        report.warn(
            "音ずれ補正が全台 ON です",
            "プライマリ1台は OFF が定石です（クロック供給元そのものなので補正できない）。",
        )
    else:
        report.fail(
            f"音ずれ補正 OFF が {len(no_drift)} 台あります"
            f"（{'、'.join(s.name for s in no_drift)}）",
            "プライマリ1台を除いて ON にしないと、音がずれていきます。",
        )

    # --- プライマリは常在デバイスか ---
    if info.subdevices:
        primary = info.subdevices[0]
        virtual_hints = ("blackhole", "soundid", "virtual", "loopback")
        if any(h in _flat(primary.name) for h in virtual_hints):
            report.ok(f"プライマリ『{primary.name}』は常に存在する仮想デバイス")
        else:
            report.warn(
                f"プライマリが物理デバイス『{primary.name or '(オフライン装置)'}』です",
                "その機器を挿さない日に束のクロックが失われます。"
                "BlackHole 2ch をプライマリにしておくと日替わりでも安全です。",
            )


def _flat(text: str) -> str:
    return "".join(text.lower().split())


def check_default_output(meeting: dict, report: Report) -> None:
    target = meeting.get("multi_output_name", "会議録音用（複数出力装置）")
    section("【3】現在の出力先")

    current = dev.default_output_name()
    print(f"  macOS の既定出力: {current or '(取得できず)'}")
    if current and "".join(target.lower().split()) in "".join(current.lower().split()):
        report.ok("既定出力が複数出力装置になっている")
    else:
        report.warn(
            "既定出力が複数出力装置ではありません",
            f"会議アプリの「スピーカー」を『{target}』に**明示的に**指定してください"
            "（Zoom/Teams はアプリ内に出力先の設定がある。"
            "ブラウザの Google Meet 等は macOS の既定出力に従うので、既定側を切り替える）。",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="会議モードのオーディオ経路を点検する")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    meeting = config.get("meeting", {})

    print("=" * 66)
    print("  会議モード オーディオ経路の点検")
    print("=" * 66)

    report = Report()
    if meeting.get("capture_backend", "screencapturekit") == "screencapturekit":
        section("【0】ScreenCaptureKit")
        try:
            from src.audio.sck_capture import has_screen_capture_permission, responsible_app_name
            if has_screen_capture_permission():
                report.ok(f"画面収録の許可あり: {responsible_app_name()}")
            else:
                report.fail(f"画面収録の許可がありません: {responsible_app_name()}", "システム設定で画面収録を許可してください。")
        except ImportError:
            report.fail("ScreenCaptureKit を import できません", "PyObjC 依存を確認してください。")
        print("  詳細診断: sck_probe.py --seconds 3")
        print("  BlackHole の点検はフォールバック用。今は使っていません。")
        print()
        print("=" * 66)
        return 0 if not report.failures else 1
    mic, loopback = check_devices(meeting, report)
    check_multi_output(meeting, mic, loopback, report)
    check_default_output(meeting, report)

    print()
    print("=" * 66)
    if report.failures:
        print(f"  {NG} 要対処 {len(report.failures)} 件 / 注意 {len(report.warnings)} 件")
        print("=" * 66)
        return 1

    if report.warnings:
        print(f"  {OK} 要対処なし（注意 {len(report.warnings)} 件）")
    else:
        print(f"  {OK} 問題なし")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
