"""会議の録音を外へ出す前の関門（会議ごとの承認・課金の確認・送った記録）。

2026-09-13 の事故を受けた歯止め。方針はリポジトリの `CLAUDE.md`、設計は `PLAN-external-stt.md`。
ここが通らなければ、音声は 1 バイトも外へ出ない。**判断できないときは送らない**（fail-closed）。

関門は 4 つ。運用者 の決定（2026-09-13）:

1. **既定はオフ** — `settings.yaml` の `meeting.external_stt.enabled` は false。
2. **会議ごとに確認する** — 設定を on にしただけでは送らない。実行のたびに画面で聞く。
   「一度 on にしたら以後ずっと」にしない。「検証だから」で通らない形にするため。
   画面が無い（TTY でない）ときは聞けない ＝ **送らない**。
3. **落ちたら手元に戻る** — 承認が取れない・オフライン・課金を確認できないときは、
   呼び出し側がローカルの文字起こしへ落ちる。会議が失われてはいけない。
4. **送った記録を残す** — いつ・どの会議・どのプロジェクトへ出したかをセッションに書く。
   09-13 の事故は「送ったこと自体が後から辿れない」のがいちばん痛かった。
"""

from __future__ import annotations

import json
import logging
import os
import re
import select
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO
from zoneinfo import ZoneInfo

SEND_LOG = "external_sends.jsonl"

logger = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")


class ExternalSendRefused(Exception):
    """関門を通らなかった。呼び出し側はローカル経路へ落ちること。"""


@dataclass
class ExternalSttConfig:
    """`settings.yaml` の `meeting.external_stt`。"""

    enabled: bool = False
    """既定は false。true にしても、会議ごとの確認は別に要る。"""

    provider: str = "gemini"
    """送り先。gemini | deepgram。2026-09-18 に Deepgram を足した（速さ 252 倍・費用 半分・
    精度は同じ。`PLAN-cloud-hybrid.md` の 8b）。どちらでも**関門は同じ数だけ通る**。"""

    deepgram_key_file: str = ""
    deepgram_model: str = "nova-3"

    billing_project: str = ""
    """送り先の Google Cloud プロジェクト ID。このシステム専用のものを 1 つ固定する
    （運用者 決定 2026-09-13 — 課金の実態が追えるようにするため）。"""

    model: str = "gemini-3.5-transcribe"
    key_file: str = ""
    glossary_path: str = ""
    """外のエンジン用の置き換え辞書。`config/glossary.yaml` は Whisper 特有の誤りを
    集めたもので、外のエンジンにはほとんど効かない（2026-09-13 実測）。"""

    timeout_sec: float = 900.0

    live: dict = field(default_factory=dict)
    """会議**中**に 30 秒刻みで送るときの刻み方（`src/stt/live_batch.py` の `LiveBatchConfig`）。
    会議中に送るかどうかは `stt.engine: gemini` で決まる。ここは刻み方だけ。"""

    auto_login: bool = True
    """起動時に gcloud のログインが切れていたら、**ブラウザでログイン画面を自動で開いて待つ**。

    2026-09-14・09-15 と続けて切れていた（Google Workspace の Cloud セッションの再認証。16 時間ほどで切れる）。
    端末で `gcloud auth login` を打つ手間を無くす（運用者 依頼「会議の前に必ず自動でログインするように」）。
    ブラウザでアカウントを選ぶ 1 クリックは残る（Google の再認証は人の操作を要求するため）。
    """

    login_timeout_sec: float = 180.0
    """ログインを待つ上限。超えたら待たずに始める（外へは出さず、手元で処理する）。"""

    ask_timeout_sec: float = 120.0
    """画面の確認を何秒待つか。超えたら「送らない」に倒す。

    会議後の作り直しは**無人で走る**（`meeting.finalize_after`）。待ち続ける実装だと、
    席を外している間ずっと止まって議事録ができない。返事が無い＝承認されていない。
    """

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ExternalSttConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (values or {}).items() if key in known})


@dataclass
class Approval:
    """関門を通った証。"""

    billing_project: str
    files: list[Path] = field(default_factory=list)
    at: str = ""


BILLING_CREDENTIAL = "~/.config/meeting-copilot/billing-status-reader.json"
"""課金の確認専用のサービスアカウントの鍵（環境変数 `MEETING_BILLING_CREDENTIAL` で差し替え可）。

人のログインは Google Workspace の再認証で 16 時間ほどで切れる（2026-09-14・09-15）。サービスアカウントは切れない。
このアカウントの権限は `resourcemanager.projects.get` だけ（課金が有効かを読むだけ・送信には使わない）。
鍵が無ければ、今までどおり人のログインで確かめる。
"""


def billing_credential() -> Path | None:
    path = Path(os.environ.get("MEETING_BILLING_CREDENTIAL") or BILLING_CREDENTIAL).expanduser()
    return path if path.is_file() and path.stat().st_size > 0 else None


def billing_enabled(project: str, *, timeout_sec: float = 60.0) -> tuple[bool, str]:
    """課金が有効か gcloud に聞く。(判定, 理由) を返す。

    判定できないとき（gcloud が無い・**認証切れ**・オフライン）は False を返す。
      無料枠は送った内容が製品改善に使われ、人間のレビュアーが読むことがあるため、
      「確認できないなら送らない」に倒す。
    `timeout_sec` は、会議前の点検から短く呼ぶためにある（起動を長く止めない）。
    確認専用のサービスアカウントの鍵があれば先にそれで聞く（ログイン切れの影響を受けない）。
      それで確かめられなければ、人のログインで聞き直す。
    """
    credential = billing_credential()
    if credential is not None:
        ok, reason = _ask_billing(project, timeout_sec, credential)
        if ok:
            return ok, reason
        logger.warning("確認専用のアカウントでは課金を確かめられませんでした（人のログインで聞き直します）: %s", reason)
    return _ask_billing(project, timeout_sec, None)


def _ask_billing(project: str, timeout_sec: float, credential: Path | None) -> tuple[bool, str]:
    env = None
    if credential is not None:
        # gcloud の「使うアカウント」は変えない（このコマンドだけ鍵で聞く）
        env = {**os.environ, "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": str(credential)}
    try:
        result = subprocess.run(
            ["gcloud", "billing", "projects", "describe", project,
             "--format", "value(billingEnabled)"],
            capture_output=True, text=True, timeout=timeout_sec, env=env,
        )
    except FileNotFoundError:
        return False, "gcloud が見つかりません"
    except subprocess.TimeoutExpired:
        return False, "gcloud の応答がありません（オフライン？）"
    if result.returncode != 0:
        detail = result.stderr.strip()
        if credential is None and _login_problem(detail):
            # これが実際に起きる（2026-09-14）。放っておくと会議中に黙って手元へ落ちる
            return False, LOGIN_EXPIRED
        return False, f"課金状態を確認できません: {_short(detail)}"
    if result.stdout.strip() != "True":
        return False, f"{project} は課金が有効ではありません"
    return True, "課金が有効です" + ("（確認専用のアカウント）" if credential is not None else "")


LOGIN_EXPIRED = "gcloud の認証が切れています（端末で `gcloud auth login`）"


def _login_problem(detail: str) -> bool:
    lowered = detail.lower()
    return any(mark in lowered for mark in (
        "reauthentication", "reauth", "gcloud auth login", "no credentialed accounts",
        "do not currently have an active account", "invalid_grant", "refresh token"))


def login_and_wait(
    project: str,
    *,
    timeout_sec: float = 180.0,
    on_status: Callable[[str], None] | None = None,
    popen=subprocess.Popen,
    check=None,
) -> tuple[bool, str]:
    """gcloud のログイン画面をブラウザで開き、終わるのを待って、課金をもう一度確かめる。

    「確認できないなら送らない」は変えない。ログインが終わらなければ False のまま返す。
    アカウントは前回と同じもの（`gcloud config get-value account`）を指定して、選ぶ手間を減らす。
    """
    say = on_status or (lambda message: None)
    check = check or (lambda: billing_enabled(project, timeout_sec=20.0))
    account = ""
    try:
        found = subprocess.run(["gcloud", "config", "get-value", "account"],
                               capture_output=True, text=True, timeout=10)
        account = found.stdout.strip() if found.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "gcloud が見つかりません"
    command = ["gcloud", "auth", "login", "--brief"] + ([account] if "@" in account else [])
    say(f"Google のログイン画面をブラウザで開きました{f'（{account}）' if account else ''}。"
        f"アカウントを選ぶと続きます（{timeout_sec:.0f} 秒待ちます）")
    try:
        process = popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return False, "gcloud が見つかりません"
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        say("ログインが終わらなかったので、待たずに進みます（外へは出しません）")
        return False, f"ログインが {timeout_sec:.0f} 秒で終わりませんでした"
    ok, reason = check()
    say("ログインできました。外へ出す準備ができています" if ok else f"ログインしましたが、まだ確認できません: {reason}")
    return ok, reason


def _short(detail: str, limit: int = 120) -> str:
    """gcloud の長いエラーを 1 行に畳む。画面に出すので、途中で切れた文のまま出さない。"""
    text = " ".join(detail.split())
    text = re.sub(r"^ERROR:\s*\([^)]*\)\s*", "", text)     # 呼び出したコマンド名の前置きは落とす
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _default_ask(prompt: str, stream: TextIO, out: TextIO, timeout_sec: float) -> bool | None:
    """画面で聞く。承認 True／断り False／**聞けなかった None**。

    聞けなかった（TTY でない）と断られたを分ける。どちらも送らないのは同じだが、
    理由が違うのに同じ文を出すと「n と答えたつもりが、実は何も聞かれていなかった」に
    気づけない（2026-09-13 に実際に紛らわしかった）。
    返事を待ち続けない。会議後の作り直しは無人で走るので、待ち続ける実装だと
    席を外している間ずっと止まって議事録ができない。時間切れは「送らない」。
    """
    if not stream.isatty():
        return None
    out.write(prompt)
    out.flush()
    try:
        if timeout_sec > 0:
            ready, _, _ = select.select([stream], [], [], timeout_sec)
            if not ready:
                out.write(f"\n  返事が無いので送りません（{timeout_sec:.0f} 秒待ちました）。\n")
                out.flush()
                return False
        answer = stream.readline()
    except (EOFError, KeyboardInterrupt, OSError, ValueError):
        return False
    return answer.strip().lower() in {"y", "yes", "はい"}


def approve(
    config: ExternalSttConfig,
    session_dir: Path,
    files: list[Path],
    *,
    ask: Callable[[str], bool] | None = None,
    stream: TextIO | None = None,
    out: TextIO | None = None,
) -> Approval:
    """4 つの関門を順に通す。通らなければ `ExternalSendRefused`。"""
    out = out or sys.stdout
    stream = stream or sys.stdin

    if not config.enabled:
        raise ExternalSendRefused(
            "外部の文字起こしは切ってあります（settings.yaml の meeting.external_stt.enabled）"
        )
    if config.provider == "gemini" and not config.billing_project:
        raise ExternalSendRefused(
            "送り先のプロジェクト ID が設定されていません（meeting.external_stt.billing_project）"
        )
    if config.provider == "deepgram" and not config.deepgram_key_file:
        raise ExternalSendRefused(
            "Deepgram のキーの場所が設定されていません（meeting.external_stt.deepgram_key_file）"
        )

    minutes = sum(audio_minutes(path) for path in files)
    prompt = (
        "\n"
        f"  ── 会議の録音を外（{'Deepgram' if config.provider == 'deepgram' else 'Google'}）へ出します ──\n"
        f"    会議   : {session_dir.name}\n"
        f"    送るもの: {'、'.join(path.name for path in files)}"
        f"{f'（約 {minutes:.0f} 分）' if minutes else ''}\n"
        f"    送り先  : {destination(config)}\n"
        "    中身はクライアントとの会話そのものです（金額・実名を含みます）。\n"
        "    出さない場合は手元の文字起こしで作ります（少し時間がかかるだけです）。\n"
        "  出してよいですか? [y/N]: "
    )
    approved = ask(prompt) if ask is not None else _default_ask(prompt, stream, out, config.ask_timeout_sec)
    if approved is None:
        raise ExternalSendRefused(
            "画面が無いので承認を聞けません（端末から実行してください）。"
            "聞けない＝承認されていない、として送りません"
        )
    if not approved:
        raise ExternalSendRefused("この会議については承認されませんでした")

    ok, reason = paid_account(config)
    if not ok:
        raise ExternalSendRefused(f"{reason}。確認できないので送りません")

    return Approval(
        # 送り先の識別子（記録に残す）。Gemini はプロジェクト ID、Deepgram は "deepgram"
        billing_project=config.billing_project or config.provider,
        files=list(files),
        at=datetime.now(JST).isoformat(timespec="seconds"),
    )


def destination(config: ExternalSttConfig) -> str:
    """送り先の名前（画面と記録に出す）。"""
    if config.provider == "deepgram":
        return f"Deepgram（{config.deepgram_model}）"
    return f"Google Cloud プロジェクト {config.billing_project}"


def paid_account(config: ExternalSttConfig) -> tuple[bool, str]:
    """**送ったものが学習に使われない口**であることを機械で確かめる（確かめられなければ送らない）。

    無料枠は送った内容が製品改善に使われ、人間が読むことがある。しかも消せない。
    だから「確かめられない」ときも送らない（fail-closed）。送り先が増えても、この関門は同じ。
    """
    if config.provider == "deepgram":
        from src.stt.deepgram_transcribe import DeepgramApi
        from src.stt.gemini_transcribe import load_key

        try:
            found = DeepgramApi(load_key(config.deepgram_key_file)).data_use()
        except Exception as error:                    # noqa: BLE001
            return False, f"Deepgram の口座を確かめられません（{error}）"
        return bool(found.get("ok")), str(found.get("why", ""))
    return billing_enabled(config.billing_project)


def record_send(session_dir: Path, approval: Approval, *, model: str, note: str = "") -> Path:
    """送ったことをセッションに残す（後から辿れるように）。"""
    path = session_dir / SEND_LOG
    entry = {
        "at": approval.at or datetime.now(JST).isoformat(timespec="seconds"),
        "session": session_dir.name,
        "billing_project": approval.billing_project,
        "model": model,
        "files": [file.name for file in approval.files],
        "note": note,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def audio_minutes(path: Path) -> float:
    """wav のヘッダから長さ（分）を読む。読めなければ 0。"""
    try:
        import wave

        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate() / 60.0
    except Exception:
        return 0.0


_audio_minutes = audio_minutes
"""旧名。新しい呼び出しは `audio_minutes` を使う。"""
