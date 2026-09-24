"""事前資料から、その会議の**固有名詞**（社名・製品名・略称）を拾う。手元だけ・LLM なし。

使い道は 3 つ。

1. **その会議だけの守り札**: 置き換え辞書が固有名詞の一部を壊さないようにする
   （09-14 の会議で、社名「MDクラウド」が「クラウド→Claude」で「MDClaude」になった）
2. **会議中の要約への手がかり**: 「表記はこれにそろえる」として毎窓のプロンプトに短く載せる
3. **議事録と辞書の候補**: 議事録の材料に並べ、誤変換を正しい表記へ寄せる手がかりにする

外の文字起こし（Gemini）には語彙を渡せない（語彙指定とタイムスタンプを同時に使えない・2026-09-13 実測）。
だから「起こす前に教える」ではなく「起こしたあとで守る・寄せる」に使う。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TERMS_FILE = "prep_terms.json"
"""セッションに残す固有名詞の一覧（会議のあとの道具が読む）。prep/ の外に置く（.json は資料として読まれるため）。"""

_KATAKANA = re.compile(r"[ァ-ヶ][ァ-ヶー・]{3,}")
_LATIN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9&.\-]*[A-Za-z0-9](?![A-Za-z0-9])")
"""`\b` は使わない。日本語の文字も単語の一部とみなされ、「TORIAの」から TORIA が取れない。"""
_NUMBERING = re.compile(r"^(?:[A-Za-z]{1,2}\d{1,3}|[A-Za-z]\d?)$")
"""Q5・No1・A1 のような番号は語ではない。"""
_QUOTED = re.compile(r"[「『]([^」』\n]{2,20})[」』]")
_HIRAGANA = re.compile(r"[ぁ-ん]")
_COMPANY = re.compile(r"(?:株式会社|合同会社|有限会社)[ 　]?([^\s、。,，:：（()）「」]{2,15})"
                      r"|([^\s、。,，:：（()）「」]{2,15})[ 　]?(?:株式会社|合同会社|有限会社)")

# どの会議にも出る一般語は固有名詞ではない（守り札にすると、辞書の正しい置き換えを止めてしまう）
_COMMON = {
    "サービス", "システム", "データ", "ユーザー", "プロジェクト", "スケジュール", "コスト", "メール", "ミーティング",
    "タスク", "チーム", "メンバー", "プラン", "オプション", "レポート", "ページ", "サイト", "ウェブサイト", "コンテンツ",
    "マーケティング", "デザイン", "イメージ", "ファイル", "フォーマット", "テンプレート", "ツール", "アカウント",
    "ログイン", "パスワード", "アップデート", "リリース", "テスト", "サポート", "サーバー", "クラウド", "アプリ",
    "アプリケーション", "ソフトウェア", "ハードウェア", "ネットワーク", "セキュリティ", "プライバシー", "ポリシー",
    "ビジネス", "マネジメント", "コミュニケーション", "フィードバック", "レビュー", "リスト", "カテゴリー", "ステータス",
    "パートナー", "クライアント", "カスタマー", "エンジニア", "ディレクター", "マネージャー", "リーダー", "スタッフ",
    "キャンペーン", "イベント", "セミナー", "ウェビナー", "インタビュー", "アンケート", "フォーム", "ボタン", "メニュー",
    "トップページ", "ダッシュボード", "ワークフロー", "プロセス", "ステップ", "フェーズ", "ゴール", "ターゲット",
    "パーセント", "ポイント", "トータル", "オンライン", "オフライン", "リアルタイム", "ドキュメント", "マニュアル",
}
_COMMON_LATIN = {
    "the", "and", "for", "with", "from", "this", "that", "are", "you", "your", "our", "not", "all", "can", "will",
    "pdf", "url", "http", "https", "www", "com", "html", "css", "api", "faq", "todo", "memo", "note", "page", "file",
    "data", "web", "app", "ok", "no", "yes", "id", "jp", "co", "inc", "ltd", "etc", "vs", "am", "pm", "q1", "q2", "q3", "q4",
}


def extract_terms(text: str, *, limit: int = 20, extra: list[str] | None = None) -> list[str]:
    """資料の本文から固有名詞らしい語を、**確かなものから順に**返す（最大 `limit`）。

    確かさの順: 「## 用語」の節 → LLM のまとめが挙げた語（`extra`）→ 会社名（株式会社〇〇）→
    大文字や数字を含む英字（TORIA・OEM）→ ひらがなを含まない短い「」→ 2 回以上出る長いカタカナ。
    カタカナは一般語が多い（実際の資料で ブランド・キーワード・ロジック が上位に来た）ので最後に回す。
    """
    first: list[str] = []
    in_terms = False
    for line in text.splitlines():
        if line.startswith("#"):
            in_terms = line.strip("# ").strip() in {"用語", "用語集"}
            continue
        if in_terms and re.match(r"^\s*[-・*]\s*\S", line):     # 箇条書きの行だけ（節の後ろの本文は語ではない）
            word = re.split(r"[:：（(]", line.strip().lstrip("-・* ").strip())[0].strip()
            word = re.sub(r"^(?:株式会社|合同会社|有限会社)[ 　]?|[ 　]?(?:株式会社|合同会社|有限会社)$", "", word)
            if word:
                first.append(word)    # 「株式会社」は外す（話し言葉では付かないので、守り札として当たらない）
    first += [str(word).strip() for word in (extra or []) if str(word).strip()]

    companies = Counter((left or right).strip() for left, right in _COMPANY.findall(text))
    latin: Counter[str] = Counter()
    for match in _LATIN.findall(text):
        if match.lower() in _COMMON_LATIN or match.isdigit() or _NUMBERING.match(match):
            continue
        if len(match) < 3:
            continue                     # AI・CX・IT は略語だが、どの会議にも出て守り札にならない
        if (any(ch.isupper() for ch in match) and len(match) >= 4) or match.isupper() \
                or any(ch.isdigit() for ch in match):
            latin[match] += 1            # TORIA・OEM・Revuno・Findly・gemma4（日本語の資料の英字は名前が多い）
    quoted = Counter(word.strip() for word in _QUOTED.findall(text)
                     if len(word.strip()) <= 12 and not _HIRAGANA.search(word))
    katakana: Counter[str] = Counter()
    for match in _KATAKANA.findall(text):
        word = match.strip("・")
        if word not in _COMMON and len(word) >= 5:
            katakana[word] += 1

    ordered = list(dict.fromkeys(first))
    for group, minimum in ((companies, 1), (latin, 1), (quoted, 1), (katakana, 2)):
        for word, count in group.most_common():
            if len(ordered) >= limit:
                return ordered[:limit]
            if count >= minimum and not any(word in known for known in ordered):   # 既に拾った語の一部は足さない
                ordered.append(word)
    return ordered[:limit]


def save_terms(session_dir: Path, terms: list[str]) -> Path:
    path = Path(session_dir) / TERMS_FILE
    path.write_text(json.dumps(terms, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_terms(session_dir: Path) -> list[str]:
    path = Path(session_dir) / TERMS_FILE
    if not path.exists():
        return []
    try:
        return [str(word) for word in json.loads(path.read_text(encoding="utf-8"))]
    except (json.JSONDecodeError, OSError):
        return []


def prompt_line(terms: list[str], *, max_chars: int = 300) -> str:
    """毎窓のプロンプトに載せる 1 節。短く（毎回載るので、長いと資料の本文を押し出す）。"""
    if not terms:
        return ""
    picked: list[str] = []
    for word in terms:
        if len("、".join(picked + [word])) > max_chars:
            break
        picked.append(word)
    return f"### この会議の固有名詞（表記はこれにそろえる）\n{'、'.join(picked)}"
