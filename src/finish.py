"""会議のあとの仕上げを、**押してから**走らせる（あとから再開できるようにする）。

2026-09-16 運用者 指示。会議が終わった瞬間に重い処理が走り始めると困る場面がある:

- 対面の商談で、終わったらすぐ移動する
- シェアオフィスや外出先で、Wi-Fi が切れる（途中で落ちて積む）
- 次の予定が詰まっていて、数分も待てない

∴ 会議の終わりは**録音と全文を確実に残すところまで**で一区切りにし、そこから先
（録音からの作り直し・議事録・辞書の候補・声の台帳）は**人が押したときだけ**走らせる。
ブラウザを閉じたあとでも、画面「会議アシスタント」の**「会議のあと」**から同じ処理を再開できる。

残っている工程は `pending.json` に書く。**何が終わっていて何が残っているかを、機械が読める形で残す**
（「たぶん終わっている」を人の記憶に任せない）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
PENDING_FILE = "pending.json"
STATE_FILE = "finish_state.json"
"""終わった工程の印（ファイルの有無だけでは分からないもの: 声の台帳・外で作った議事録）。"""
LOG_FILE = "finish.log"

FINISH_MODES = ("ask", "auto", "later")
"""ask=会議の終わりに聞く（既定）／auto=そのまま仕上げる／later=いつも後回し。"""


@dataclass
class Step:
    key: str
    label: str
    minutes: str
    """おおよその所要（画面に出す。「数分かかる」と分かっていれば人は判断できる）。"""


STEPS = [
    Step("finalize", "録音から全文を作り直す", "2〜4 分"),
    Step("voices", "話した人の声を台帳に覚える", "数秒"),
    Step("glossary", "置き換え辞書の候補を出す", "数秒"),
    Step("minutes", "議事録を作る", "1〜3 分"),
    Step("audio", "録音を mp3 に畳む（マイク別＋統合）", "1〜2 分"),
]


def mark(session_dir: Path, key: str, note: str = "", url: str = "") -> dict:
    """工程が終わった印を残す。「声を覚えた」「議事録は Notion にある」はファイルでは分からない。"""
    path = Path(session_dir) / STATE_FILE
    state = read_state(session_dir)
    state[key] = {"at": datetime.now(JST).isoformat(timespec="seconds"), "note": note, "url": url}
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state[key]


def read_state(session_dir: Path) -> dict:
    path = Path(session_dir) / STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def done_steps(session_dir: Path, *, voice_library: Path | None = None,
               voice_enabled: bool = True) -> set[str]:
    """もう終わっている工程。ファイルの有無＋終わった印で見る（記憶では判断しない）。

    `voice_enabled=False`（声の台帳を使わない設定）なら、声を覚える工程は**最初から終わり**。
      そうしないと、台帳に誰も入らない＝いつまでも「仕上げが残っています」と出続ける
      （2026-09-20 の配布版の受け入れ確認で発見。配布版の既定は off なので、**全員がこれを踏む**）。
    """
    session_dir = Path(session_dir)
    found = set(read_state(session_dir))
    if not voice_enabled:
        found.add("voices")
    if voice_library is not None and _voices_learned(voice_library, session_dir.name):
        found.add("voices")
    if (session_dir / "transcripts_final.jsonl").exists():
        found.add("finalize")
    if (session_dir / "glossary_candidates.json").exists():
        found.add("glossary")
    if (session_dir / "minutes.md").exists() or (session_dir / "minutes_prompt.md").exists():
        found.add("minutes")
    audio = session_dir / "audio"
    if audio.is_dir() and any(audio.glob("*.mp3")):
        found.add("audio")
    if not (session_dir / "recording_remote.wav").exists() and not (session_dir / "recording_self.wav").exists():
        found.add("audio")      # 録音が無い会議（リプレイ等）では畳むものが無い
    return found


def _voices_learned(voice_library: Path, session: str) -> bool:
    """声の台帳に、その会議で覚えた人がいるか。台帳が正本（セッション側に印が無くても分かる）。"""
    path = Path(voice_library)
    if not path.exists():
        return False
    try:
        people = json.loads(path.read_text(encoding="utf-8")).get("people") or {}
    except (json.JSONDecodeError, OSError):
        return False
    return any(entry.get("session") == session for entries in people.values() for entry in entries)


def remaining(session_dir: Path, *, wanted: list[str] | None = None,
              voice_library: Path | None = None, voice_enabled: bool = True) -> list[Step]:
    """まだ残っている工程。"""
    done = done_steps(session_dir, voice_library=voice_library, voice_enabled=voice_enabled)
    keys = set(wanted) if wanted is not None else {step.key for step in STEPS}
    return [step for step in STEPS if step.key in keys and step.key not in done]


def write_pending(session_dir: Path, steps: list[Step], *, note: str = "") -> Path:
    """残りをファイルに残す。あとで画面が拾って「仕上げる」を出す。"""
    path = Path(session_dir) / PENDING_FILE
    path.write_text(json.dumps({
        "at": datetime.now(JST).isoformat(timespec="seconds"),
        "note": note,
        "steps": [{"key": step.key, "label": step.label, "minutes": step.minutes} for step in steps],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_pending(session_dir: Path) -> dict | None:
    path = Path(session_dir) / PENDING_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_pending(session_dir: Path) -> None:
    """残りが無くなったら消す（退避は不要 — 中身は残りの一覧だけで、成果物ではない）。"""
    path = Path(session_dir) / PENDING_FILE
    if path.exists():
        path.unlink()


def saved_already(session_dir: Path) -> list[str]:
    """会議の終わりに「ここまでは確実に残っています」と言える中身。"""
    session_dir = Path(session_dir)
    found = []
    for name, label in (("recording_remote.wav", "相手側の録音"), ("recording_self.wav", "自分の録音"),
                        ("transcripts.jsonl", "会議中の全文"), ("minutes_input.md", "議事録の材料"),
                        ("state.json", "決定事項・TODO")):
        path = session_dir / name
        if path.exists() and path.stat().st_size > 0:
            found.append(label)
    return found
