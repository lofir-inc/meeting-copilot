"""ほかのアプリの負荷待ちのテスト（実プロセスは見ない）。"""

from src.system_load import app_cpu_percent, wait_until_quiet

PS = """  PID  %CPU COMM
  101  38.2 /Applications/MacWhisper.app/Contents/MacOS/MacWhisper
  102   6.8 /Applications/MacWhisper.app/Contents/MacOS/MacWhisperHelper
  103  44.7 /Applications/zoom.us.app/Contents/MacOS/zoom.us
  104   0.0 /usr/sbin/cupsd
"""


def test_cpu_of_one_app_sums_its_processes():
    """1 アプリが複数プロセスに分かれるので合計で見る。"""
    assert app_cpu_percent(["MacWhisper"], PS) == 45.0


def test_other_apps_are_not_counted():
    assert app_cpu_percent(["zoom"], PS) == 44.7
    assert app_cpu_percent(["Illustrator"], PS) == 0.0


def test_broken_output_is_ignored():
    assert app_cpu_percent(["MacWhisper"], "") == 0.0
    assert app_cpu_percent(["MacWhisper"], "PID %CPU COMM\nこわれた行") == 0.0


def test_quiet_from_the_start_needs_one_confirmation():
    """瞬間的な谷で誤判定しないよう、静かでも 1 回だけ確認してから始める。"""
    slept: list[float] = []
    notes: list[tuple[float, float]] = []
    result = wait_until_quiet(["MacWhisper"], snapshot=lambda: "PID %CPU COMM\n 1 1.0 MacWhisper",
                              sleep=slept.append, now=lambda: 0.0, quiet_samples=2, poll_sec=15.0,
                              on_wait=lambda cpu, waited: notes.append((cpu, waited)))
    assert result["reason"] == "落ち着いた"
    assert slept == [15.0]
    assert notes == []          # 静かなときは「待機中」と出さない


def test_waits_until_the_other_app_finishes():
    samples = ["PID %CPU COMM\n 1 90.0 MacWhisper"] * 3 + ["PID %CPU COMM\n 1 2.0 MacWhisper"] * 3
    clock = iter([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0])
    notes: list[tuple[float, float]] = []
    result = wait_until_quiet(["MacWhisper"], snapshot=lambda: samples.pop(0), sleep=lambda _s: None,
                              now=lambda: next(clock), quiet_samples=2, on_wait=lambda cpu, waited: notes.append((cpu, waited)))
    assert result["reason"] == "落ち着いた"
    assert len(notes) == 3          # 忙しい間だけ知らせる


def test_gives_up_after_the_timeout():
    """待ち続けて何も出ないより、上限で先へ進む。"""
    clock = iter([0.0, 100.0, 200.0, 300.0])
    result = wait_until_quiet(["MacWhisper"], snapshot=lambda: "PID %CPU COMM\n 1 90.0 MacWhisper",
                              sleep=lambda _s: None, now=lambda: next(clock), timeout_sec=120.0)
    assert result["reason"] == "待ち時間の上限"


def test_no_apps_means_no_waiting():
    assert wait_until_quiet([], snapshot=lambda: "")["reason"] == "無効"
