"""議事録のあとの連鎖 — 哲学カードと製品 Fact（議事録の連鎖 の Step 4・5）。

抽出は LLM（Claude CLI）、登録は決定論の共有ライブラリ。分け方は `task_hub-minutes` スキルと同じ。

ここで守ること（どちらも、間違えると**台帳が汚れて後片付けが重い**）:

1. **哲学カードはペルソナを解決できなければ登録しない**（fail-close）。誤ったペルソナで書くと
   dedup（Title+Persona）が効かず、同じカードが増え続ける
2. **話者ごとに行き先が違う**。自分（自社の人間）の哲学は**自社の DB**、相手の哲学は**相手の DB**
3. **製品 Fact の主語は相手の会社**。相手が「支援会社から受けている」ものは製品ではない
   （プロンプトで縛り、ここでも会社名を渡す）
4. 抽出できない・解決できないときは**黙って飛ばさず**、理由を結果に残して画面へ出す
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CARD_CATEGORIES = {"判断軸", "価値観", "行動原則", "ビジョン", "思考パターン"}
FACT_TYPES = {"spec", "price", "use_case", "narrative", "pain_point", "differentiation", "other"}
MAX_CARDS = 5
MAX_FACTS = 8


@dataclass
class ChainResult:
    """連鎖の結果（画面にそのまま出す）。"""

    cards: int = 0
    cards_skipped: int = 0
    facts: int = 0
    facts_skipped: int = 0
    notes: list[str] = field(default_factory=list)
    """走らなかった理由・書けなかったもの。黙って飛ばさない。"""

    def as_dict(self) -> dict:
        return {"cards": self.cards, "cards_skipped": self.cards_skipped,
                "facts": self.facts, "facts_skipped": self.facts_skipped, "notes": self.notes}


def extract_json(text: str) -> list[dict]:
    """LLM の返事から JSON の配列を取り出す。取れなければ空（落とさない）。"""
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text[text.find("["):text.rfind("]") + 1] if "[" in text else ""
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def clean_cards(rows: list[dict], *, speakers: set[str]) -> list[dict]:
    """哲学カードの形を整える。不明話者のものと、形の違うものは捨てる。"""
    found = []
    for row in rows[:MAX_CARDS * 3]:
        speaker = str(row.get("speaker", "")).strip()
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()
        if not (title and content) or "不明話者" in speaker:
            continue
        if speakers and speaker not in speakers:   # いない人の哲学は作らない
            continue
        category = str(row.get("category", "")).strip()
        found.append({
            "speaker": speaker, "title": title[:40], "content": content[:400],
            "category": category if category in CARD_CATEGORIES else "価値観",
            "sensitivity": str(row.get("sensitivity", "Medium")).strip() or "Medium",
            "tags": [str(tag).strip() for tag in (row.get("tags") or []) if str(tag).strip()][:5],
        })
        if len(found) >= MAX_CARDS:
            break
    return found


def clean_facts(rows: list[dict], *, rejected: list[str]) -> list[dict]:
    """製品 Fact の形を整える。却下済みの名前は落とす（ネガティブメモリ）。"""
    blocked = {str(name).strip() for name in rejected if str(name).strip()}
    found = []
    for row in rows[:MAX_FACTS * 3]:
        name = str(row.get("product_name", "")).strip()
        key = str(row.get("key", "")).strip()
        value = str(row.get("value", "")).strip()
        if not (name and key and value):
            continue
        if any(word and (word in name or name in word) for word in blocked):
            continue
        fact_type = str(row.get("fact_type", "")).strip()
        found.append({
            "product_name": name[:100], "key": key[:50], "value": value[:500],
            "fact_type": fact_type if fact_type in FACT_TYPES else "other",
            "confidence": str(row.get("confidence", "medium")).strip() or "medium",
        })
        if len(found) >= MAX_FACTS:
            break
    return found


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def find_person(notion, people_db: str, speaker: str) -> dict | None:
    """話者ラベルから People Master の人を引く（`Speaker Labels` の multi_select）。

    **完全一致を先に見る。** 連携先の突き合わせは双方向の部分一致を先着で採るので、
    「自分」が「自分花子」（家族）に当たる。実データで踏んだ（2026-09-17）:

        Hanako Sato  … ['はなこ', '花子', 'Hanako', '家族', '自分花子', '自分 花子']
        Taro Sato… ['自社自分', '自分', '自分太郎', '自分 太郎', 'Taro']

    「自分」は Taro の**ラベルそのもの**なので、完全一致を先に見れば当たる。
    部分一致しか無く、それが**複数の人**に当たるときは選ばない（誰の哲学か分からないものは書かない）。
    """
    if not (people_db and speaker):
        return None
    target = _norm(speaker)
    exact, partial = [], []
    for row in notion.query_database(people_db):
        props = row.get("properties", {})
        labels = [_norm(name) for name in notion.prop_multi_select(props, "Speaker Labels")]
        organization = [value["id"] for value in (props.get("Organization", {}) or {}).get("relation", [])]
        entry = {"id": row["id"], "name": notion.prop_title(props),
                 "internal": bool((props.get("Is Internal", {}) or {}).get("checkbox", False)),
                 "organization": organization[0] if organization else ""}
        if any(label == target for label in labels if label):
            exact.append(entry)
        elif any(label and (target in label or label in target) for label in labels):
            partial.append(entry)
    if len(exact) == 1:
        return exact[0]
    if exact:
        return {"ambiguous": [entry["name"] for entry in exact]}
    if len(partial) == 1:
        return partial[0]
    if partial:
        return {"ambiguous": [entry["name"] for entry in partial]}
    return None


def client_of_person(notion, context, person: dict, cache: dict) -> dict | None:
    """その人の**所属先**（People Master の `Organization`）をクライアントとして引く。

    商流をまたぐ人がいる（さくら歯科 ⇒ GX ⇒ 自社、取引先A ⇒ GX ⇒ 自社）。
    GX の人の哲学は、会議の相手（さくら歯科）ではなく **GX の DB** に入れたい
    （運用者 依頼 2026-09-17）。会議ごとの相手ではなく、**人の所属**で行き先を決める。
    """
    org_id = str(person.get("organization") or "")
    if not org_id:
        return None
    if org_id in cache:
        return cache[org_id]
    try:
        props = notion.get_page(org_id).get("properties", {})
        name = notion.prop_title(props, "client_name") or notion.prop_title(props)
        client = context.resolve_client(name) if name else None
    except Exception:                               # noqa: BLE001 - 引けなければ会議の相手に落とす
        logger.warning("所属先を引けませんでした: %s", org_id, exc_info=True)
        client = None
    cache[org_id] = client
    return client


def personas_of(notion, persona_db: str, people_master_id: str) -> list[dict]:
    """その人に紐づくペルソナ（複数可。連携先は名簿のリレーションで引く）。"""
    if not (persona_db and people_master_id):
        return []
    rows = notion.query_database(persona_db, filter={
        "property": "People Master", "relation": {"contains": people_master_id}}, page_size=10)
    return [{"page_id": row["id"], "name": notion.prop_title(row.get("properties", {}))} for row in rows]


def _ask(llm, prompt: str, body: str) -> list[dict]:
    """LLM に投げて JSON の配列をもらう。失敗しても連鎖を止めない。"""
    try:
        return extract_json(llm.generate_text(f"{prompt}\n\n## 会議の全文\n{body}"))
    except Exception as exc:                        # noqa: BLE001 - CLI 側の失敗は理由だけ残す
        logger.warning("抽出に失敗しました: %s", exc)
        raise


def philosophy(session_dir: Path, *, llm, prompts: Path, transcript: str, speakers: set[str],
               minutes_page_id: str, minutes_url: str, client: dict, self_client: dict | None,
               self_name: str, context, philosophy_lib, task_hub_minutes_lib, notion,
               model: str = "", personas: dict | None = None, people_db: str = "") -> ChainResult:
    """哲学カードを抽出して登録し、議事録に話者別の一覧を足す（WF の Step 4）。

    行き先の決め方は **連携先と同じ 2 段**（2026-09-17 に 議事録の連鎖 /
    哲学カードの抽出 を読んで合わせた）:

      話者ラベル → People Master の `Speaker Labels`（multi_select）で人を引く
                → Persona Master を `People Master`（relation）contains で引く（複数可）

    自社の人かどうかも People Master の `Is Internal` で決まる（話者名の一致に頼らない）。
    さらに、その人の **`Organization`（所属先）** を見て、**会議の相手ではなく所属先の DB** へ入れる。
    商流をまたぐ人がいるため（さくら歯科 ⇒ GX ⇒ 自社。GX の人の哲学は GX の DB へ）。
    **1 人に複数のペルソナ**が付くときは、連携先と同じく **1 枚のカードの Persona に全員を入れる**
    （同じ洞察を人数分コピーしない）。dedup は 1 人目で効く。

    `personas`（`settings.yaml` の `task_hub.personas`）は**上書きと逃げ道**。People Master に
    ラベルが無い人や、急いで別のペルソナに入れたいときに、名前で直接引く。

    `resolve_persona` は部分一致なので、**別人に当たっても黙って登録される**。
    当たった名前を結果に残して、目で確かめられるようにする。
    """
    result = ChainResult()
    rows = _ask(llm, (prompts / "philosophy_cards.md").read_text(encoding="utf-8"), transcript)
    cards = clean_cards(rows, speakers=speakers)
    if not cards:
        result.notes.append("哲学カードは見つかりませんでした")
        return result

    groups: list[dict] = []
    org_cache: dict = {}
    for speaker in dict.fromkeys(card["speaker"] for card in cards):
        mine = [card for card in cards if card["speaker"] == speaker]
        # 1 段目: People Master の「Speaker Labels」で人を引く（連携先と同じ道筋）。
        #   自社の人かどうかも、その人の `Is Internal` で決まる（話者名の一致に頼らない）
        person_row = find_person(notion, people_db, speaker) if people_db else None
        if person_row and person_row.get("ambiguous"):
            # 誰の哲学か決められないものは書かない（家族や同姓に当たることがある）
            result.notes.append(f"{speaker}: People Master で {len(person_row['ambiguous'])} 人に当たりました"
                                f"（{'、'.join(person_row['ambiguous'])}）。"
                                f"Speaker Labels を分けるか、画面の対応表で指名してください")
            person_row = None
        internal = person_row["internal"] if person_row else (bool(self_client) and speaker == self_name)
        # 1 段目の半: その人の所属先（商流をまたぐ人は、会議の相手ではなく所属先の DB へ）
        home = client_of_person(notion, context, person_row, org_cache) if person_row else None
        target = home or (self_client if (internal and self_client) else client)
        cards_db = str((target or {}).get("philosophy_cards_db_id") or "")
        persona_db = str((target or {}).get("persona_db_id") or "")

        people: list[tuple[str, dict]] = []
        missing: list[str] = []
        if person_row:
            # 2 段目: その人に紐づくペルソナ（複数可）を relation で引く
            people = [(entry["name"], entry) for entry in personas_of(notion, persona_db, person_row["id"])]
            if not people:
                missing = [f"{person_row['name']} に紐づくペルソナ"]
        # 手元の対応表は上書き／逃げ道（People Master に無い人・急ぐとき）
        override = [str(name) for name in (personas or {}).get(speaker, [])]
        if override or not people:
            wanted = override or [speaker]
            found = [(name, context.resolve_persona(persona_db, name)) for name in wanted] if persona_db else []
            if any(person for _, person in found):
                people = [(name, person) for name, person in found if person]
                missing = [name for name, person in found if not person]
            elif not people:
                missing = [name for name, _ in found] or wanted
        if not (cards_db and people):
            # fail-close（誤ったペルソナで書くと dedup が効かず重複を量産する）
            result.notes.append(f"{speaker}: ペルソナを解決できず"
                                f"（{'、'.join(missing) or persona_db or 'DB 未設定'}）、"
                                f"哲学カード {len(mine)} 枚を登録しませんでした")
            result.cards_skipped += len(mine)
            continue
        if missing:                                 # 解決できた分だけ書く。書けなかった名前は残す
            result.notes.append(f"{speaker}: ペルソナ「{'、'.join(missing)}」は見つかりませんでした")
        person = people[0][1]
        payload = [{key: card[key] for key in ("title", "content", "category", "sensitivity", "tags")}
                   for card in mine]
        try:
            registered = philosophy_lib.register_philosophy_cards(
                payload, cards_db_id=cards_db, persona_id=person["page_id"],
                source_url=minutes_url, source_type="議事録",
                llm=notion.llm_label(model, skill="realtime-minutes") if model else None)
        except Exception as exc:                    # noqa: BLE001
            result.notes.append(f"{speaker}: 哲学カードを登録できませんでした（{exc}）")
            result.cards_skipped += len(mine)
            continue
        made = registered.get("registered") or []
        result.cards += len(made)
        result.cards_skipped += int(registered.get("skipped") or 0)
        # 2 人目からは、作ったカードの Persona に足す（1 枚のカードを複数のペルソナに紐づける）
        if len(people) > 1 and made:
            ids = [person["page_id"] for _, person in people]
            for card in made:
                if not notion.update_page_safe(card.get("id"), cards_db,
                                               {"Persona": notion.relation_val(ids)}):
                    result.notes.append(f"{speaker}: 2 人目以降のペルソナを紐づけられませんでした")
                    break
        # どのペルソナに入ったかを残す（部分一致なので、別人に当たっていないか目で見る）
        names = "、".join(person.get("name", name) for name, person in people)
        where = str((target or {}).get("client_name") or ("自社" if target is self_client else "相手"))
        result.notes.append(f"{speaker} → ペルソナ「{names}」（{where} の DB）に {len(mine)} 枚")
        groups.append({"speaker": speaker, "titles": [card["title"] for card in mine]})

    if groups:
        try:                                        # 一覧は入らなくても登録は残す
            task_hub_minutes_lib.append_philosophy_summary(minutes_page_id, groups)
        except Exception:                           # noqa: BLE001
            logger.warning("哲学カードの一覧を議事録へ足せませんでした", exc_info=True)
            result.notes.append("哲学カードの一覧は議事録に足せませんでした（登録は済んでいます）")
    return result


def product_facts(session_dir: Path, *, llm, prompts: Path, transcript: str, minutes_page_id: str,
                  client: dict, client_name: str, task_hub_minutes_lib) -> ChainResult:
    """製品 Fact を抽出して登録する（WF の Step 5）。"""
    result = ChainResult()
    facts_db = str(client.get("product_facts_db_id") or "")
    master_db = str(client.get("product_service_db_id") or "")
    if not facts_db:
        result.notes.append(f"「{client_name}」に製品 Fact の DB が無いので、製品 Fact は取りませんでした")
        return result

    try:
        rejected = [str(name) for name in (task_hub_minutes_lib.fetch_rejected_products(facts_db) or [])]
    except Exception:                               # noqa: BLE001
        rejected = []
    prompt = (prompts / "product_facts.md").read_text(encoding="utf-8")
    prompt = prompt.replace("{client_name}", client_name).replace(
        "{rejected}", "、".join(rejected) if rejected else "（無し）")
    facts = clean_facts(_ask(llm, prompt, transcript), rejected=rejected)
    if not facts:
        result.notes.append("製品 Fact は見つかりませんでした")
        return result
    try:
        registered = task_hub_minutes_lib.register_product_facts(facts_db, master_db, facts,
                                                          page_id=minutes_page_id, source_type="議事録")
    except Exception as exc:                        # noqa: BLE001
        result.notes.append(f"製品 Fact を登録できませんでした（{exc}）")
        result.facts_skipped += len(facts)
        return result
    result.facts = len(registered.get("created") or [])
    result.facts_skipped = int(registered.get("skipped") or 0)
    return result
