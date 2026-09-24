"""ほかのアプリの負荷を見て、重い処理を始めるタイミングを待つ。

なぜ要るか（2026-09-11 の実会議で実測）:
MacWhisper は会議が終わると自動で文字起こしを始める（止める設定は無い。2026-09-12 に設定と
plist の両方を確認）。その最中に GPU リセットが 2 回起き、こちらの文字起こしと Ollama が落ちた。
MacWhisper には運用実績があるので外さない（運用者 判断 2026-09-12）。
∴ **こちらが待つ**。会議後の作り直しは、相手の処理が落ち着いてから始める。
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _ps_snapshot() -> str:
    """いま動いているプロセスの CPU 使用率一覧を返す（失敗したら空）。"""
    try:
        result = subprocess.run(["ps", "-Ao", "pid,%cpu,comm"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        logger.debug("ps を実行できませんでした", exc_info=True)
        return ""
    return result.stdout if result.returncode == 0 else ""


def app_cpu_percent(names: list[str], snapshot: str) -> float:
    """指定したアプリ名を含むプロセスの CPU 使用率を合計して返す。

    1 アプリが複数プロセスに分かれる（ヘルパーを持つ）ので合計で見る。
    """
    total = 0.0
    for line in snapshot.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        _, cpu, command = parts
        if any(name.lower() in command.lower() for name in names):
            try:
                total += float(cpu)
            except ValueError:
                continue
    return total


def wait_until_quiet(
    names: list[str],
    *,
    threshold_percent: float = 20.0,
    timeout_sec: float = 1800.0,
    poll_sec: float = 15.0,
    quiet_samples: int = 2,
    snapshot: Callable[[], str] = _ps_snapshot,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    on_wait: Callable[[float, float], None] | None = None,
) -> dict:
    """指定アプリの CPU が閾値を下回るまで待つ。返り値は待った結果の記録。

    - `quiet_samples` 回続けて静かなら「落ち着いた」とみなす（瞬間的な谷で誤判定しない）
    - `timeout_sec` を超えたら待つのをやめて先へ進む（待ち続けて何も出ないより、動くほうがよい）
    """
    if not names or timeout_sec <= 0:
        return {"waited_sec": 0.0, "reason": "無効"}
    started = now()
    quiet = 0
    while True:
        cpu = app_cpu_percent(names, snapshot())
        if cpu < threshold_percent:
            quiet += 1
            if quiet >= quiet_samples:
                return {"waited_sec": round(now() - started, 1), "reason": "落ち着いた", "cpu": cpu}
        else:
            quiet = 0
        waited = now() - started
        if waited >= timeout_sec:
            return {"waited_sec": round(waited, 1), "reason": "待ち時間の上限", "cpu": cpu}
        # 知らせるのは相手が忙しいときだけ（静かなのに「待機中」と出ると紛らわしい）
        if on_wait is not None and cpu >= threshold_percent:
            on_wait(cpu, waited)
        sleep(poll_sec)
