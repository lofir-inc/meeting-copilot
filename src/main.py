"""エントリポイント — インタビュー支援システムの起動。"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

from src.app.user_path import ensure_user_path
from src.audio import ffmpeg
from src.orchestrator import Orchestrator

MODE_INTERVIEW = "interview_stereo"
MODE_MEETING = "meeting_loopback"


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def resolve_session_dir(prep_dir: Path) -> Path:
    """prep ディレクトリ内の .md ファイル名からセッション名を決定する。"""
    md_files = sorted(prep_dir.glob("*.md"))
    if md_files:
        # 最初の .md ファイルの拡張子なし名をセッション名にする
        session_name = md_files[0].stem
    else:
        from datetime import datetime
        session_name = datetime.now().strftime("%Y-%m-%d_session")
    return Path("workspace/sessions") / session_name


def stale_prep_files(prep_dir: Path, today: date) -> list[tuple[str, int]]:
    """今日より前に更新された事前資料を (ファイル名, 何日前) で返す。"""
    stale: list[tuple[str, int]] = []
    for md_file in sorted(prep_dir.glob("*.md")):
        days = (today - datetime.fromtimestamp(md_file.stat().st_mtime).date()).days
        if days > 0:
            stale.append((md_file.name, days))
    return stale


def stale_prep_warnings(stale: list[tuple[str, int]], prep_dir: Path) -> list[str]:
    """古い事前資料の警告文。ターミナルだけでなく、起動セルフチェックの画面にも出す。

    なぜ要るか（2026-09-13 に判明）: `workspace/prep/` は**会議をまたぐ共有の 1 フォルダ**で、
    中身は 4 月の別の会議の資料のままだった。全セッションがそれをコピーするので、09-11 の定例も
    4 月のインタビュー計画を評価基準にしていた（画面に出た「問い 21 件」は無関係な会議のもの）。
    ターミナルには前から出していたが、運用者 は画面を見て会議をするので気づけない。
    """
    if not stale:
        return []
    names = "、".join(f"{name}（{days} 日前）" for name, days in stale)
    return [f"事前資料が今日のものではありません: {names}。"
            f"このまま始めると、要約と問いはこの資料を基準にします（入れ替えは {prep_dir}）"]


def setup_session(session_dir: Path, prep_dir: Path | None) -> None:
    """セッションディレクトリを作成し、prep ファイルをコピーする。

    `prep_dir` が None なら**何もコピーしない**（会議モードの既定＝毎回リセット）。
    資料は画面から足す（`MeetingOrchestrator.attach_prep`）。
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    session_prep = session_dir / "prep"
    session_prep.mkdir(exist_ok=True)
    if prep_dir is None:
        return

    for md_file in sorted(prep_dir.glob("*.md")):
        dest = session_prep / md_file.name
        if not dest.exists():
            shutil.copy2(md_file, dest)


def main() -> None:
    ensure_user_path()  # .app・launchd から起こされても claude / ollama / ffmpeg が見える
    parser = argparse.ArgumentParser(description="ニアリアルタイム・インタビュー支援システム")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="設定ファイルのパス (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--prep-dir",
        default=None,
        help="事前準備資料のディレクトリ (default: workspace/prep)。"
             "会議モードは既定で使わない（毎回リセットし、資料は画面から足す）。"
             "明示して渡したときだけ、会議モードでもコピーする",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="セッション名を明示指定（省略時は prep ファイル名から自動決定）",
    )
    parser.add_argument(
        "--mode",
        choices=[MODE_INTERVIEW, MODE_MEETING],
        default=None,
        help="動作モード（省略時は settings.yaml の mode を使う）",
    )
    parser.add_argument(
        "--client",
        default=None,
        help="この会議のクライアント（会議ごとに画面でも選べる。議事録とタスクの行き先）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="DEBUG ログを有効にする",
    )
    args = parser.parse_args()

    # 設定ファイル読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"設定ファイルが見つかりません: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    mode = args.mode or config.get("mode", MODE_INTERVIEW)

    # セッションディレクトリ決定
    prep_dir = Path(args.prep_dir) if args.prep_dir else Path("workspace/prep")
    if args.session:
        session_dir = Path("workspace/sessions") / args.session
    elif mode == MODE_MEETING:
        # 会議は prep のファイル名で決めない。prep に古い資料が残っていると、今日の会議が
        #   その日のフォルダに入る（2026-09-11 本番: 4 月の 2026-04-11-toria に入った）
        session_dir = Path("workspace/sessions") / datetime.now().strftime("%Y-%m-%d_%H%M")
    else:
        session_dir = resolve_session_dir(prep_dir)

    # 会議モードは既定で「毎回リセット」。共有フォルダの資料を使い回すと、別の会議の計画が
    #   そのまま評価基準になる（2026-09-11 の定例は 4 月のインタビュー計画を基準にしていた）。
    #   資料は画面（ダッシュボードの「事前資料」）から、会議の途中でも足せる。
    reset_prep = (mode == MODE_MEETING
                  and config.get("meeting", {}).get("prep_reset", True)
                  and not args.prep_dir)
    setup_session(session_dir, None if reset_prep else prep_dir)
    startup_warnings: list[str] = []
    if reset_prep:
        print("\n  事前資料は付けずに始めます（要るものは画面の「事前資料」から足せます）。")
    if mode == MODE_MEETING and not reset_prep:
        stale = stale_prep_files(prep_dir, datetime.now().date())
        startup_warnings = stale_prep_warnings(stale, prep_dir)
        if stale:
            print("\n  ⚠ 事前資料が今日のものではありません（要約の材料として読み込まれます）:")
            for name, days in stale:
                print(f"      {name}（{days} 日前）")
            print(f"    不要なら {prep_dir} から外してから起動し直してください。\n")

    # output の workspace_dir をセッションディレクトリに上書き
    config.setdefault("output", {})["workspace_dir"] = str(session_dir)

    # ログファイルもセッション内に保存
    setup_logging(args.verbose, log_file=session_dir / "interview.log")
    logger = logging.getLogger(__name__)

    # 対面モードの入力デバイスも名前で解決する（番号は接続機器で毎回ずれる）。
    #   AudioConfig は device_candidates を知らないので、ここで取り除いてから渡す。
    #   Orchestrator 自体には手を入れない＝対面モードの経路は不変。
    audio_cfg = config.setdefault("audio", {})
    device_candidates = audio_cfg.pop("device_candidates", None)
    if mode == MODE_INTERVIEW and device_candidates:
        from src.audio.devices import DeviceNotFoundError as _DNF
        from src.audio.devices import resolve_input_device

        try:
            resolved = resolve_input_device(device_candidates, label="インタビュー用マイク")
            audio_cfg["device"] = resolved.index
            logger.info("入力デバイスを名前で解決: %s", resolved)
        except _DNF as exc:
            # 候補が見つからないときは設定の device 番号にフォールバックする。
            # 対面モードは L/R キャリブレーションで気付けるので、ここでは止めない。
            logger.warning("%s", exc)
            logger.warning(
                "settings.yaml の audio.device=%s をそのまま使います。"
                "意図した機器か確認してください。",
                audio_cfg.get("device"),
            )

    # 会議中は ffmpeg を使わない（録音は wav）。止めずに、終わったあと困ることだけ先に言う。
    ffmpeg.warn_if_missing()

    logger.info("=== リアルタイム支援システム起動 ===")
    logger.info("Mode: %s", mode)
    logger.info("Config: %s", config_path)
    logger.info("Session: %s", session_dir)

    # オーケストレーター起動 — モードで分岐する。
    # 対面（interview_stereo）の経路は従来どおり Orchestrator をそのまま使う。
    if mode == MODE_MEETING:
        from src.audio.devices import DeviceNotFoundError
        from src.meeting_orchestrator import MeetingOrchestrator

        orch = MeetingOrchestrator(config, session_dir)
        if args.client:
            # 「この会議は誰の会議か」を先に決めておく（画面でも変えられる）
            orch.choose_client(args.client, how="起動時の --client")
        orch.startup_warnings = startup_warnings   # 起動セルフチェックの画面に出す
        orch.load_prep_materials(session_dir / "prep")
        try:
            orch.run()
        except DeviceNotFoundError as exc:
            # 黙って別デバイスを掴まず、候補と実在デバイスを見せて止まる
            logger.error("デバイスを解決できませんでした")
            print(f"\n{exc}\n", file=sys.stderr)
            sys.exit(2)
    else:
        orch = Orchestrator(config)
        orch.load_prep_materials(session_dir / "prep")
        orch.run()


if __name__ == "__main__":
    main()
