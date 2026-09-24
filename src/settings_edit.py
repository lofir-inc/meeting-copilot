"""設定（`settings.yaml`）を画面から直す（機能の ON/OFF・よく触る値・ペルソナの対応表）。

`yaml.dump` で書き直さない。この設定ファイルはコメント（なぜその値にしたか・実測の経緯）が
中身の半分なので、行で差し込む／抜く（置き換え辞書と同じ考え方＝`src/text/glossary_edit.py`）。

直す前のファイルは `workspace/99-trash/settings/` に退避してから書く（直接上書きしない）。
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

JST = ZoneInfo("Asia/Tokyo")
SECTION = "task_hub"
KEY = "personas"
HEADER = """  personas:                     # 会議の話者名 → Notion のペルソナ名（哲学カードの行き先）
                                # 画面「会議アシスタント」→「声の台帳」から直せる
"""


def read_personas(path: Path) -> dict[str, list[str]]:
    """いまの対応表。1 人に**複数**のペルソナを紐づけられる（連携先の設定に合わせる）。

    読めなければ空（画面を止めない）。1 つだけ書いてあるときも list にして返す。
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    table = ((data.get(SECTION) or {}).get(KEY)) or {}
    found: dict[str, list[str]] = {}
    for speaker, value in table.items():
        names = [str(name).strip() for name in (value if isinstance(value, list) else [value])]
        names = [name for name in names if name]
        if names:
            found[str(speaker)] = names
    return found


def split_names(value: str | list) -> list[str]:
    """画面から来た「自社社長、自社開発者」を分ける（読点・カンマ・改行のどれでも）。"""
    if isinstance(value, list):
        return [str(name).strip() for name in value if str(name).strip()]
    return [name.strip() for name in re.split(r"[、,\n]+", str(value)) if name.strip()]


def set_persona(path: Path, speaker: str, persona: str | list, *,
                trash_dir: Path | None = None) -> dict[str, list[str]]:
    """1 人ぶん足す／直す／外す（空なら外す）。複数のペルソナを並べられる。"""
    speaker = speaker.strip()
    names = split_names(persona)
    if not speaker:
        raise ValueError("話者名を入れてください")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定がありません: {path}")
    if trash_dir is not None:
        _backup(path, Path(trash_dir))

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start, end = _bounds(lines)
    # 複数は 1 行に並べる（`自分: [自社社長, 自社開発者]`）。読みやすさを優先
    written = _quote(names[0]) if len(names) == 1 else "[" + ", ".join(_quote(name) for name in names) + "]"
    entry = f"    {_quote(speaker)}: {written}\n"

    if start is None:                       # 節そのものが無い → `task_hub:` の直下に作る
        if not names:
            return read_personas(path)
        head = _section_line(lines)
        if head is None:
            raise ValueError(f"`{SECTION}:` の節が見つかりません: {path}")
        lines[head + 1:head + 1] = [HEADER, entry]
    else:
        kept = [line for line in lines[start:end] if _line_speaker(line) != speaker]
        if names:
            kept.append(entry)
        lines[start:end] = kept
    path.write_text("".join(lines), encoding="utf-8")
    yaml.safe_load(path.read_text(encoding="utf-8"))   # 壊れた YAML を残さない
    return read_personas(path)


def _quote(value: str) -> str:
    """YAML に置ける形にする（記号や空白を含む名前でも壊さない）。"""
    return value if re.fullmatch(r"[\w一-龥ぁ-んァ-ヶー々〆〤]+", value) else '"' + value.replace('"', '\\"') + '"'


def _section_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if re.match(rf"^{SECTION}:\s*$", line):
            return index
    return None


def _bounds(lines: list[str]) -> tuple[int | None, int]:
    """`personas:` の中身の範囲（見出しの次の行 〜 節の終わり）。"""
    head = _section_line(lines)
    if head is None:
        return None, 0
    for index in range(head + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith((" ", "\t")):     # 次のトップレベル
            break
        if re.match(rf"^\s{{2}}{KEY}:\s*(#.*)?$", line):
            for tail in range(index + 1, len(lines) + 1):
                if tail == len(lines):
                    return index + 1, tail
                nxt = lines[tail]
                if not nxt.strip():
                    continue
                if not re.match(r"^\s{4,}", nxt):                 # 4 字下げの行だけが中身
                    return index + 1, tail
    return None, 0


def _line_speaker(line: str) -> str | None:
    """`    自分: 自社社長` の行から話者名を取る（コメント行は None）。"""
    match = re.match(r'^\s{4,}(?:"([^"]+)"|\'([^\']+)\'|([^:#\s][^:#]*?))\s*:', line)
    if not match:
        return None
    return (match.group(1) or match.group(2) or match.group(3) or "").strip()


def _backup(path: Path, trash_dir: Path) -> Path:
    """直す前の設定を退避する（直接上書きしない）。"""
    trash_dir.mkdir(parents=True, exist_ok=True)
    target = trash_dir / f"{datetime.now(JST):%Y-%m-%d_%H%M%S}_{path.name}"
    shutil.copy2(path, target)
    return target


# --------------------------------------------------------------- 機能の ON/OFF と、よく触る値

FIELDS: list[dict] = [
    {"key": "meeting.screen_capture.enabled", "kind": "bool", "label": "画面共有を控える",
     "help": "会議アプリの窓だけを数秒ごとに見て、変わったときだけ残す。URL とページ名を手元で読む。"
             "ブラウザは会議の目印（Google Meet 等）が無ければ撮らない。外へは何も送らない"},
    {"key": "meeting.screen_capture.interval_sec", "kind": "number", "label": "　画面を見る間隔（秒）",
     "help": "短くすると細かく残るが枚数が増える（上限は max_shots）"},
    {"key": "meeting.screen_capture.ocr", "kind": "bool", "label": "　画面の文字を読む",
     "help": "切ると画像だけ残す（URL とページ名は控えない）"},
    {"key": "stt.engine", "kind": "choice", "choices": ["local", "gemini", "deepgram"],
     "label": "会議中の文字起こし", "danger": True,
     "help": "local=手元の Whisper（外へ出ない・画面に出るまで時間がかかる）／"
             "gemini・deepgram=外へ音声を出す（会議ごとに承認と、学習に使われない口かの確認を通る）。"
             "精度は三者ほぼ同じ（2026-09-18 実測）。違うのは速さと費用"},
    {"key": "meeting.external_stt.provider", "kind": "choice", "choices": ["gemini", "deepgram"],
     "label": "　外へ出すときの送り先", "danger": True,
     "help": "deepgram=速い・安い（252 倍速・¥39/時間）／gemini=27 倍速・¥83/時間"},
    {"key": "llm.engine", "kind": "choice", "choices": ["local", "gemini"],
     "label": "会議中の要約", "danger": True,
     "help": "local=手元の Ollama（外へ出ない・GPU を使う）／gemini=外へテキストを出す。"
             "実費の 57% はここ（2026-09-17 の実測）"},
    {"key": "llm.final_pass", "kind": "choice", "choices": ["none", "claude_cli"],
     "label": "会議の終わりの読み直し",
     "help": "claude_cli=全文をもう一度読んで状態を整える（サブスクの範囲・テキストのみ）"},
    {"key": "meeting.voice_library.enabled", "kind": "bool", "label": "声の台帳を使う",
     "help": "名前を付けた人の声を覚え、次の会議で「〇〇さん？」と候補を出す。"
             "声紋は個人を識別できる情報。この Mac の中だけに置く"},
    {"key": "meeting.external_stt.enabled", "kind": "bool", "label": "会議中の文字起こしを外（Gemini）で回す",
     "danger": True,
     "help": "会議の音声を外へ出す。on にしても会議ごとに画面で聞き、課金が有効なプロジェクトを"
             "機械で確かめてからでないと送らない。断れば手元の Whisper に落ちる"},
    {"key": "meeting.finish_mode", "kind": "choice", "choices": ["ask", "auto", "later"],
     "label": "会議のあとの仕上げ",
     "help": "ask=画面で聞く（既定）／auto=すぐ走らせる／later=いつも後回し"},
    {"key": "meeting.finalize_after", "kind": "bool", "label": "録音から全文を作り直す",
     "help": "会議中の文字起こしより正確になる（65 分の会議で 2〜3 分）"},
    {"key": "meeting.open_folder_after", "kind": "bool", "label": "終わったらフォルダを開く",
     "help": "Finder で成果物のフォルダを開く"},
    {"key": "meeting.verify_engine", "kind": "choice", "choices": ["auto", "claude_cli", "gemini", "none"],
     "label": "裏取り（🔍）の手段",
     "help": "auto=Claude CLI →（承認した会議なら）Gemini → なし"},
    {"key": "meeting.state.interval_sec", "kind": "choice", "choices": ["30", "60", "120", "180"],
     "label": "要約を作り直す間隔（秒）",
     "note": "右カラムの論点・TODO を作り直す間隔。2026-09-18 実測: 30 秒だと 58% が空振りで、"
             "1 回の送信の 82% は状態と指示の再送。延ばしても要約の中身は減らない"},
    {"key": "meeting.external_stt.live.window_sec", "kind": "choice",
     "choices": ["auto", "5", "8", "15", "30"], "label": "字幕を出す間隔（秒）", "external": True,
     "note": "auto＝エンジンに合わせる（Deepgram 8 秒＝画面まで約 9.5 秒／Gemini 30 秒＝約 43 秒）。"
             "2026-09-18 実測: Deepgram は 8 秒でも 30 秒と同じ精度（39.8%）。5 秒から崩れる（43.2%）"},
    {"key": "meeting.self_name", "kind": "text", "label": "自分の名前（字幕のラベル）",
     "help": "自分のマイクの発言に付く固定の名前"},
    {"key": "meeting.silence_alert_sec", "kind": "number", "label": "無音の警告（秒）",
     "help": "本編に入ってからこの秒数音が来なければ画面で知らせる（0 で切る）"},
    {"key": "minutes.engine", "kind": "choice", "choices": ["auto", "claude_cli", "gemini", "local", "none"],
     "label": "議事録を作る手段",
     "help": "auto=Claude CLI →（承認した会議なら）Gemini → 手元の LLM → 貼り付け用の指示書"},
    {"key": "task_hub.notion_tasks", "kind": "bool", "label": "TODO を Notion に登録する",
     "help": "会議中に「着手」を押した TODO を Tasks DB へ"},
    {"key": "task_hub.slack", "kind": "bool", "label": "チャットに知らせる",
     "help": "会議の終わりに、クライアントのチャットへ投げる"},
    {"key": "task_hub.dictionary_sync", "kind": "bool", "label": "クライアント辞書とつなぐ",
     "help": "会議の終わりに読んでキャッシュし、選んだ候補を候補として送る"},
]
"""画面から直せる設定。ここに挙げたものだけ（全部を画面に出すと、触ってはいけない値まで触れてしまう）。"""


def read_fields(path: Path) -> list[dict]:
    """いまの値を添えて返す。設定に無い項目は出さない（環境によって節が無いことがある）。"""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    except yaml.YAMLError:
        data = {}
    found = []
    for field_ in FIELDS:
        value, missing = data, False
        for part in field_["key"].split("."):
            if not isinstance(value, dict) or part not in value:
                missing = True
                break
            value = value[part]
        if missing:
            continue
        found.append({**field_, "value": value})
    return found


def set_value(path: Path, key: str, value, *, trash_dir: Path | None = None):
    """`meeting.screen_capture.enabled` のような入れ子のキーを 1 つ直す。

    行で書き換える（`yaml.dump` で書き直さない）。コメント（なぜその値にしたか）が
    この設定ファイルの中身の半分なので、消さない。
    """
    field_ = next((one for one in FIELDS if one["key"] == key), None)
    if field_ is None:
        raise ValueError(f"画面から直せる設定ではありません: {key}")
    written = _as_yaml(field_, value)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定がありません: {path}")
    if trash_dir is not None:
        _backup(path, Path(trash_dir))

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    index = _line_of(lines, key.split("."))
    if index is None:
        raise KeyError(f"設定に見つかりません: {key}")
    original = lines[index].rstrip("\n")
    head, _, tail = original.partition(":")
    at = tail.index("#") + len(head) + 1 if "#" in tail else -1
    comment = original[at:] if at >= 0 else ""
    body = f"{head}: {written}"
    # コメントの桁をそろえたまま書き戻す（読みやすさが設定ファイルの価値の半分）
    lines[index] = (body.ljust(at) + comment if at > len(body) else
                    f"{body}  {comment}" if comment else body) + "\n"
    path.write_text("".join(lines), encoding="utf-8")
    yaml.safe_load(path.read_text(encoding="utf-8"))     # 壊れた YAML を残さない
    return next((one["value"] for one in read_fields(path) if one["key"] == key), None)


def _as_yaml(field_: dict, value) -> str:
    """画面から来た値を YAML の書き方にする。"""
    kind = field_["kind"]
    if kind == "bool":
        return "true" if value in (True, "true", "True", 1, "1", "on") else "false"
    if kind == "number":
        number = float(value)
        return str(int(number)) if number == int(number) else str(number)
    text = str(value).strip()
    if kind == "choice" and text not in (field_.get("choices") or []):
        raise ValueError(f"選べるのは {'／'.join(field_.get('choices') or [])} です")
    if not text:
        raise ValueError("値を入れてください")
    return _quote(text)


def _line_of(lines: list[str], parts: list[str]) -> int | None:
    """入れ子のキーの行を探す（字下げで親子を見る）。"""
    depth, start = 0, 0
    for level, part in enumerate(parts):
        indent = level * 2
        found = None
        for index in range(start, len(lines)):
            line = lines[index]
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            current = len(line) - len(line.lstrip())
            if level and current <= depth and index > start:
                break                                   # 親の節を出た
            if current == indent and re.match(rf"^\s{{{indent}}}{re.escape(part)}\s*:", line):
                found = index
                break
        if found is None:
            return None
        depth, start = indent, found + 1
        if level == len(parts) - 1:
            return found
    return None


# --------------------------------------------------------------- 動かし方（プリセット）

PRESETS: list[dict] = [
    {"name": "secret", "label": "機密優先", "needs": ["local_stt", "local_llm"],
     "cost": "¥0", "note": "外へ何も出さない。画面に出るまで時間がかかり、要約の質も落ちる",
     "values": {"stt.engine": "local", "llm.engine": "local", "llm.final_pass": "none",
                "minutes.engine": "local", "meeting.verify_engine": "none",
                "meeting.external_stt.enabled": False}},
    {"name": "cost", "label": "コスト優先", "needs": ["local_stt", "local_llm", "claude_cli"],
     "cost": "¥0", "note": "従量課金ゼロ（サブスクの範囲）。会議中は手元の GPU を使い切る",
     "values": {"stt.engine": "local", "llm.engine": "local", "llm.final_pass": "claude_cli",
                "minutes.engine": "claude_cli", "meeting.verify_engine": "claude_cli",
                "meeting.external_stt.enabled": False}},
    {"name": "balanced", "label": "バランス", "needs": ["deepgram_key", "local_llm", "claude_cli"],
     "cost": "月 ¥250 ほど", "note": "音声だけ外へ（Deepgram・252 倍速）。要約は手元、議事録はサブスク",
     "values": {"stt.engine": "deepgram", "llm.engine": "local", "llm.final_pass": "claude_cli",
                "minutes.engine": "auto", "meeting.verify_engine": "auto",
                "meeting.external_stt.provider": "deepgram",
                "meeting.external_stt.enabled": True}},
    {"name": "quality", "label": "速さ優先", "needs": ["deepgram_key", "gemini_key"],
     "cost": "月 ¥900 ほど", "note": "文字起こしは Deepgram（速い）、要約は Gemini（賢い）。両方へ出る",
     "values": {"stt.engine": "deepgram", "llm.engine": "gemini", "llm.final_pass": "claude_cli",
                "minutes.engine": "auto", "meeting.verify_engine": "auto",
                "meeting.external_stt.provider": "deepgram",
                "meeting.external_stt.enabled": True}},
]
"""1 つ選べば、処理ごとのエンジンがまとめて決まる。費用は 2026-09-17 の実測（月 15 時間）。"""

NEED_LABELS = {
    "local_stt": "手元の文字起こし（Apple Silicon）",
    "local_llm": "手元の LLM（Ollama とモデル）",
    "claude_cli": "Claude CLI（サブスク）",
    "gemini_key": "Gemini の API キー",
    "deepgram_key": "Deepgram の API キー",
}


def presets_for(capability) -> list[dict]:
    """この Mac で使えるプリセット（使えないものは理由つきで返す）。"""
    found = []
    for preset in PRESETS:
        missing = [NEED_LABELS.get(need, need) for need in preset["needs"]
                   if not getattr(capability, need, False)]
        found.append({**preset, "usable": not missing, "missing": missing})
    return found


def current_preset(path: Path) -> str:
    """いまの設定がどのプリセットと同じか（どれとも違えば空）。"""
    values = {field_["key"]: field_["value"] for field_ in read_fields(path)}
    for preset in PRESETS:
        if all(values.get(key) == value for key, value in preset["values"].items() if key in values):
            return preset["name"]
    return ""


def apply_preset(path: Path, name: str, capability, *, trash_dir: Path | None = None) -> dict:
    """プリセットを当てる。できない機種では当てない（会議中に足りないと分かるのを避ける）。"""
    preset = next((one for one in PRESETS if one["name"] == name), None)
    if preset is None:
        raise ValueError(f"知らない動かし方です: {name}")
    missing = [NEED_LABELS.get(need, need) for need in preset["needs"]
               if not getattr(capability, need, False)]
    if missing:
        raise ValueError(f"この Mac では「{preset['label']}」を選べません（{'、'.join(missing)}が要ります）")
    if trash_dir is not None:
        _backup(Path(path), Path(trash_dir))
    changed = {}
    for key, value in preset["values"].items():
        try:
            changed[key] = set_value(path, key, value)
        except KeyError:
            continue                                # その設定を持たない環境では飛ばす
    return {"preset": name, "label": preset["label"], "changed": changed}
