"""会議のあとに、置き換え辞書の**候補**を出す。辞書へ入れるのは人が選んだものだけ。

候補の出どころは 2 つ。

1. **知っている語との近さ（LLM なし・手元）**: 参加者の名前（画面で付けた名前・声の台帳）、事前資料の
   「## 用語」、辞書の置き換え先・守り札を「正しい表記」とみなし、全文の中の**近いが違う表記**を拾う
   （例: 正しい表記「エムディクラウド」に対して、全文の「エムデイクラウド」）。
2. **議事録を作った LLM の指摘**: 議事録と同じ 1 回の呼び出しで、誤変換らしい語を挙げさせる
   （全文を読み直すための追加の呼び出しはしない）。LLM の指摘は、**全文に実際に出てくる語**だけ残す。

なぜ自動で辞書に入れないか: 置き換えは会議中の全行に効く。1 語の誤りが「クラウドファンディング」を
「Claudeファンディング」にする（2026-09-13 に実際に踏みかけた）。

候補は `glossary_candidates.md`（チェックボックス）に書き、`scripts/glossary_candidates.py <会議> --accept`
で**チェックしたものだけ**をエンジンの辞書へ足す。
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
CANDIDATES_FILE = "glossary_candidates.md"
CANDIDATES_JSON = "glossary_candidates.json"
"""画面（会議コックピット）が読む形。md は人がファイルで見るとき用。"""
LLM_FILE = "glossary_llm.json"
"""議事録を作った LLM が挙げた誤変換の候補（`src/minutes.py` が書く）。"""

LLM_MARK = "<!-- 誤変換候補 -->"
"""議事録の末尾に、この印に続けて JSON で候補を書かせる（`prompts/minutes.md`）。"""

_TOKEN = re.compile(r"[ァ-ヶー・]{3,}|[A-Za-z][A-Za-z0-9.\-]{2,}|[一-龥々]{2,}")
_CHECKED = re.compile(r"^- \[[xX]\] 「(?P<wrong>.+?)」→「(?P<right>.+?)」")
_CHECKED_PROTECT = re.compile(r"^- \[[xX]\] 守り札「(?P<word>.+?)」")
_LATIN = re.compile(r"^[A-Za-z0-9.\-]+$")


@dataclass
class Candidate:
    wrong: str
    right: str
    """置き換え先。`kind == "protect"` のときは空で、`wrong` が守り札に足す語。"""
    count: int
    source: str
    reason: str = ""
    examples: list[str] = field(default_factory=list)
    kind: str = "replace"
    """replace（置き換え）| protect（守り札に足す）"""


def fold(text: str) -> str:
    """比べるための形にそろえる（全角半角・カタカナとひらがな・空白と中黒）。"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)
    return re.sub(r"[\s・･]", "", text)


def known_terms(*, names: list[str], prep_text: str, glossary_pairs: list[tuple[str, str]],
                protected: list[str]) -> set[str]:
    """「正しい表記」とみなす語。名前は「会社 名前」「NEXIS参加者D」のように付くので、ばらして使う。"""
    terms: set[str] = set()
    for name in names:
        terms.update(token for token in _TOKEN.findall(name))
    in_terms = False
    for line in prep_text.splitlines():
        if line.startswith("## "):
            in_terms = line.strip() == "## 用語"
            continue
        if in_terms and line.strip():
            word = re.split(r"[:：（(]", line.strip().lstrip("-・* ").strip())[0].strip()
            if word:
                terms.add(word)
    terms.update(right for _, right in glossary_pairs)
    terms.update(protected)
    return {term for term in terms if len(fold(term)) >= 3}


def near_misses(rows: list[dict], terms: set[str], *, min_ratio: float = 0.8,
                existing: set[str] | None = None) -> list[Candidate]:
    """全文の中から、知っている語に**近いが違う**表記を拾う。"""
    existing = existing or set()
    folded_terms = {fold(term): term for term in terms}
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        text = str(row.get("text", ""))
        for token in set(_TOKEN.findall(text)):
            key = fold(token)
            if key in folded_terms or token in existing or len(key) < 3:
                continue
            latin = bool(_LATIN.match(key))
            if latin and len(key) < 5:
                continue            # 短い英字は別の語に近いだけ（FTP と SFTP、Web と WebP）
            best, best_ratio = None, 0.0
            for folded, term in folded_terms.items():
                if abs(len(folded) - len(key)) > 2 or key in folded or folded in key:
                    continue        # 片方がもう片方を含むのは「別の語」（Code と Codex、mini と Gemini）
                ratio = difflib.SequenceMatcher(None, key, folded).ratio()
                if ratio > best_ratio:
                    best, best_ratio = term, ratio
            if best is not None and best_ratio >= (max(min_ratio, 0.88) if latin else min_ratio):
                pair = (token, best)
                counts[pair] += text.count(token)
                examples.setdefault(pair, [])
                if len(examples[pair]) < 2:
                    examples[pair].append(text[:60])
    return [Candidate(wrong, right, count, "知っている語に近い", examples=examples[(wrong, right)])
            for (wrong, right), count in counts.most_common()]


def glossary_damage(rows: list[dict], pairs: list[tuple[str, str]], protected: list[str]) -> list[Candidate]:
    """辞書の置き換えが、**別の語の一部を壊した**跡を拾う（守り札の候補）。

    2026-09-14 の会議で、社名「MDクラウド」が辞書の「クラウド→Claude」で「MDClaude」になっていた。
    置き換え先（英字）の前後に英数字がくっついていたら、元は 1 つの語だった疑いがある。
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    examples: dict[tuple[str, str, str], list[str]] = {}
    correct = {fold(value) for _, value in pairs} | {fold(word) for word in protected}
    for wrong, right in pairs:
        if not _LATIN.match(right.replace(" ", "")) or _LATIN.match(wrong):
            continue
        pattern = re.compile(rf"[A-Za-z0-9]+{re.escape(right)}[A-Za-z0-9]*|{re.escape(right)}[A-Za-z0-9]+")
        for row in rows:
            text = str(row.get("text", ""))
            for token in pattern.findall(text):
                original = token.replace(right, wrong)
                if original in protected or fold(token) in correct:
                    continue        # 辞書が正しい表記として知っている語（GitHub）は壊れていない
                key = (original, wrong, right)
                counts[key] += 1
                examples.setdefault(key, [])
                if len(examples[key]) < 2:
                    examples[key].append(text[:60])
    return [Candidate(original, "", count, "辞書が壊した跡", f"「{wrong}→{right}」が「{original.replace(wrong, right)}」を作っている",
                      examples[(original, wrong, right)], kind="protect")
            for (original, wrong, right), count in counts.most_common()]


def from_llm(items: list[dict], rows: list[dict], *, existing: set[str] | None = None) -> list[Candidate]:
    """LLM が挙げた候補のうち、**全文に実際に出てくる**ものだけ残す（出てこない語は作り話）。"""
    existing = existing or set()
    text = "\n".join(str(row.get("text", "")) for row in rows)
    found: list[Candidate] = []
    for item in items:
        wrong = str(item.get("wrong", "")).strip()
        right = str(item.get("right", "")).strip()
        if not wrong or not right or wrong == right or wrong in existing:
            continue
        count = text.count(wrong)
        if count == 0:
            continue
        example = next((str(row.get("text", ""))[:60] for row in rows if wrong in str(row.get("text", ""))), "")
        found.append(Candidate(wrong, right, count, "議事録の LLM", str(item.get("reason", ""))[:80],
                               [example] if example else []))
    return found


def split_minutes(text: str) -> tuple[str, list[dict]]:
    """議事録の本文と、末尾に書かせた誤変換の候補（JSON）を分ける。壊れていたら候補は空にする。"""
    body, mark, tail = text.partition(LLM_MARK)
    if not mark:
        return text, []
    match = re.search(r"\[.*\]", tail, re.DOTALL)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        items = []
    return body.rstrip(), [item for item in items if isinstance(item, dict)]


def merge(*groups: list[Candidate]) -> list[Candidate]:
    """同じ置き換えは 1 つにまとめる（出どころは並べる）。"""
    merged: dict[tuple[str, str, str], Candidate] = {}
    for group in groups:
        for candidate in group:
            key = (candidate.kind, candidate.wrong, candidate.right)
            if key in merged:
                current = merged[key]
                if candidate.source not in current.source:
                    current.source += f"・{candidate.source}"
                current.reason = current.reason or candidate.reason
            else:
                merged[key] = candidate
    return sorted(merged.values(), key=lambda item: item.count, reverse=True)


def write_candidates(session_dir: Path, candidates: list[Candidate], glossary_path: Path) -> Path:
    path = Path(session_dir) / CANDIDATES_FILE
    lines = [
        f"# 置き換え辞書の候補（{Path(session_dir).name}）",
        "",
        f"辞書に入れたいものに `x` を付けて、`python scripts/glossary_candidates.py {Path(session_dir)} --accept` を実行すると、",
        f"**チェックしたものだけ** `{glossary_path}` に足します。",
        "置き換えは会議中の全行に効きます。ほかの語の一部を壊さないか（「クラウド」→「クラウドファンディング」）を見てから選んでください。",
        "辞書に向くのは**何度も同じ形で出る**語（社名・製品名・人名）です。1 回だけの言い間違い・文脈の直し"
        "（「経団連」→「経営層」のような普通の語どうし）は、別の会議で正しい語を壊すので入れないでください。",
        "",
    ]
    if not candidates:
        lines.append("（候補はありません）")
    for candidate in candidates:
        reason = f" — {candidate.reason}" if candidate.reason else ""
        if candidate.kind == "protect":
            lines.append(f"- [ ] 守り札「{candidate.wrong}」を足す（{candidate.count} 回・{candidate.source}）{reason}")
        else:
            lines.append(f"- [ ] 「{candidate.wrong}」→「{candidate.right}」（{candidate.count} 回・{candidate.source}）{reason}")
        for example in candidate.examples:
            lines.append(f"  - 例: {example}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def checked(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """チェックの付いた (置き換え, 守り札) を返す。"""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    pairs = [(match["wrong"], match["right"]) for line in lines if (match := _CHECKED.match(line))]
    guards = [match["word"] for line in lines if (match := _CHECKED_PROTECT.match(line))]
    return pairs, guards


def append_to_glossary(glossary_path: Path, pairs: list[tuple[str, str]], session: str,
                       guards: list[str] | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """辞書に足す（置き換えは `replacements:`、守り札は `protect:` の末尾）。

    コメント（なぜその語を入れたか）を残すため、YAML を書き直さず行で差し込む。既にある語は足さない。
    """
    import yaml

    glossary_path = Path(glossary_path)
    text = glossary_path.read_text(encoding="utf-8") if glossary_path.exists() else ""
    data = yaml.safe_load(text) or {}
    table = data.get("replacements") or {}
    known_guards = set(data.get("protect") or [])
    added = [(wrong, right) for wrong, right in pairs if wrong not in table]
    added_guards = [word for word in dict.fromkeys(guards or []) if word not in known_guards]
    note = f"  # ── 会議のあとの候補から選んだもの（{session}・{datetime.now(JST):%Y-%m-%d}）"
    # 同じ会議・同じ日の見出しが既にあれば、その下に足す（1 語ごとに見出しが増えない）
    if added:
        text = _append_to_section(text, "replacements", [
            f"  {json.dumps(wrong, ensure_ascii=False)}: {json.dumps(right, ensure_ascii=False)}" for wrong, right in added],
            note=note)
    if added_guards:
        text = _append_to_section(text, "protect", [
            f"  - {json.dumps(word, ensure_ascii=False)}" for word in added_guards], note=note)
    if added or added_guards:
        yaml.safe_load(text)          # 壊れた YAML を書かない（読めなければここで落ちる）
        glossary_path.write_text(text, encoding="utf-8")
    return added, added_guards


def _append_to_section(text: str, key: str, lines: list[str], *, note: str = "") -> str:
    """トップレベルの `key:` の節に行を足す。同じ見出し（note）が**その節の中に**あれば、その直下へ。

    節をまたいで見出しを探すと、守り札の行が置き換えの節に入って YAML が壊れる（2026-09-16 に踏んだ）。
    """
    header = re.search(rf"^{key}:[^\n]*$", text, re.MULTILINE)
    if header is None:
        return text.rstrip("\n") + f"\n\n{key}:\n" + "\n".join(([note] if note else []) + lines) + "\n"
    following = re.search(r"^[A-Za-z_][\w-]*:", text[header.end():], re.MULTILINE)
    cut = header.end() + following.start() if following else len(text)
    # 「その節だけ」を切り出す（節の手前まで、ではない。前は protect の行が replacements に入った）
    head, section, rest = text[:header.end()], text[header.end():cut], text[cut:]
    if note and note in section:
        section = section.replace(note + "\n", note + "\n" + "\n".join(lines) + "\n", 1)
        return head + section + rest
    insertion = "\n".join(([note] if note else []) + lines)
    section = section.rstrip("\n") + "\n\n" + insertion + "\n"
    return head + section + ("\n" + rest if following else "")


def glossary_for_session(session_dir: Path, meeting: dict, repo: Path) -> Path:
    """その会議の文字起こしに使ったエンジンの辞書。Whisper と Gemini で誤り方が違うので辞書も別。"""
    sends = Path(session_dir) / "external_sends.jsonl"
    live = sends.exists() and any('"live_batch"' in line for line in sends.read_text(encoding="utf-8").splitlines())
    external = (meeting.get("external_stt") or {}).get("glossary_path") or ""
    chosen = external if live and external else meeting.get("glossary_path", "config/glossary.yaml")
    path = Path(chosen).expanduser()
    return path if path.is_absolute() else repo / path


def build_for_session(session_dir: Path, *, glossary_path: Path, self_name: str,
                      extra_names: list[str] | None = None,
                      task_hub_pairs: list[dict] | None = None) -> tuple[Path, list[Candidate]]:
    """会議 1 本ぶんの候補を作って `glossary_candidates.md` に書く。"""
    from src.text.glossary import load_glossary, load_protected

    session_dir = Path(session_dir)
    rows_path = session_dir / "transcripts_final.jsonl"
    if not rows_path.exists():
        rows_path = session_dir / "transcripts.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if rows_path.exists() else []
    names = {str(row.get("speaker", "")) for row in rows}
    renames = session_dir / "renames.jsonl"
    if renames.exists():
        names.update(json.loads(line)["new"] for line in renames.read_text(encoding="utf-8").splitlines() if line.strip())
    names = [name for name in names | set(extra_names or []) if name and name != self_name and not name.startswith("不明話者")]
    prep_dir = session_dir / "prep"
    prep_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                          for path in sorted(prep_dir.glob("*.md"))) if prep_dir.is_dir() else ""
    from src.text import task_hub_dictionary

    pairs = load_glossary(glossary_path)
    protected = load_protected(glossary_path)
    # タスク管理 の辞書に既にある誤り（候補・除外も）は出さない。タスク管理 が知っている正しい表記は手がかりにする
    existing = {wrong for wrong, _ in pairs} | task_hub_dictionary.known_wrongs(task_hub_pairs or [])
    names = names + task_hub_dictionary.correct_notations(task_hub_pairs or [])
    from src.text.terms import load_terms

    terms = known_terms(names=names + load_terms(session_dir), prep_text=prep_text,
                        glossary_pairs=pairs, protected=protected)
    llm_path = session_dir / LLM_FILE
    llm_items = json.loads(llm_path.read_text(encoding="utf-8")) if llm_path.exists() else []
    candidates = merge(glossary_damage(rows, pairs, protected),
                       from_llm(llm_items, rows, existing=existing),
                       near_misses(rows, terms, existing=existing))
    from src.text.glossary_edit import decided

    done = decided(session_dir)      # 画面で入れた／見送った候補は、作り直しても出さない
    candidates = [c for c in candidates if (c.kind, c.wrong) not in done]
    (session_dir / CANDIDATES_JSON).write_text(json.dumps(
        [{"kind": c.kind, "wrong": c.wrong, "right": c.right, "count": c.count, "source": c.source,
          "reason": c.reason, "examples": c.examples} for c in candidates], ensure_ascii=False, indent=1), encoding="utf-8")
    return write_candidates(session_dir, candidates, glossary_path), candidates
