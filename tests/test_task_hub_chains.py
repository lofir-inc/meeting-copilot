"""議事録のあとの連鎖 — 哲学カードと製品 Fact（議事録の連鎖 の Step 4・5）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import task_hub_chains

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
TRANSCRIPT = "自分: 会議のあとに読み直すのが大事です\n山田: うちは安さでは勝負しません"


class Llm:
    """Claude CLI の代わり。渡された返事を順に出す。"""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else "[]"


class Philosophy:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def register_philosophy_cards(self, cards, *, cards_db_id, persona_id, source_url, source_type, llm=None):
        if self.fail:
            raise RuntimeError("DB が違います")
        self.calls.append({"cards": cards, "db": cards_db_id, "persona": persona_id})
        return {"registered": [{"title": card["title"], "url": "u"} for card in cards], "skipped": 0}


class Minutes:
    def __init__(self, rejected: list[str] | None = None) -> None:
        self.rejected = rejected or []
        self.summary: list[dict] = []
        self.facts: list[dict] = []

    def append_philosophy_summary(self, page_id, groups):
        self.summary = groups
        return {"groups": len(groups)}

    def fetch_rejected_products(self, facts_db_id):
        return self.rejected

    def register_product_facts(self, facts_db, master_db, facts, *, page_id, source_type):
        self.facts = facts
        return {"created": [{"title": fact["product_name"]} for fact in facts], "skipped": 1, "cleaned": len(facts)}


class Context:
    def __init__(self, personas: dict[str, str] | None = None) -> None:
        self.personas = personas or {}

    def resolve_persona(self, persona_db_id, name):
        page_id = self.personas.get(name)
        return {"page_id": page_id, "name": name} if page_id else None


class Notion:
    @staticmethod
    def llm_label(model, skill=None):
        return f"{model} ({skill})"

    @staticmethod
    def relation_val(page_ids):
        return {"relation": [{"id": value} for value in page_ids]}


CARDS_JSON = json.dumps([
    {"speaker": "山田", "title": "安さでは勝負しない", "content": "値段ではなく品質で選ばれたい。" * 3,
     "category": "価値観", "sensitivity": "Low", "tags": ["価格"]},
    {"speaker": "自分", "title": "あとで読み直す", "content": "その場の要約は下書き。" * 3,
     "category": "行動原則", "sensitivity": "Low", "tags": []},
], ensure_ascii=False)


class TestClean:
    def test_不明話者といない人の哲学は作らない(self):
        rows = [{"speaker": "不明話者3", "title": "あ", "content": "い" * 60},
                {"speaker": "誰か", "title": "う", "content": "え" * 60},
                {"speaker": "山田", "title": "安さでは勝負しない", "content": "お" * 60}]

        cards = task_hub_chains.clean_cards(rows, speakers={"山田", "自分"})

        assert [card["title"] for card in cards] == ["安さでは勝負しない"]

    def test_知らない分類は価値観に寄せる(self):
        rows = [{"speaker": "山田", "title": "あ", "content": "い" * 60, "category": "なにか"}]

        assert task_hub_chains.clean_cards(rows, speakers={"山田"})[0]["category"] == "価値観"

    def test_却下済みの製品名は拾わない(self):
        """ネガティブメモリ（過去に却下したものを毎回出さない）。"""
        rows = [{"product_name": "テント", "key": "耐水圧", "value": "20000mm"},
                {"product_name": "AI 記事制作", "key": "内容", "value": "支援を受けている"}]

        facts = task_hub_chains.clean_facts(rows, rejected=["AI 記事制作"])

        assert [fact["product_name"] for fact in facts] == ["テント"]

    def test_JSONを囲みから取り出す(self):
        text = "はい。\n```json\n[{\"a\": 1}]\n```\n"

        assert task_hub_chains.extract_json(text) == [{"a": 1}]


class TestPhilosophy:
    def _run(self, *, context, philosophy, minutes, self_client=None, llm=None):
        return task_hub_chains.philosophy(
            Path("/tmp"), llm=llm or Llm(CARDS_JSON), prompts=PROMPTS, transcript=TRANSCRIPT,
            speakers={"山田", "自分"}, minutes_page_id="page-1", minutes_url="https://notion.so/page-1",
            client={"philosophy_cards_db_id": "cards-client", "persona_db_id": "persona-client"},
            self_client=self_client, self_name="自分", context=context, philosophy_lib=philosophy,
            task_hub_minutes_lib=minutes, notion=Notion(), model="claude-opus-5")

    def test_話者ごとに行き先を分ける(self):
        """自分の哲学は自社の DB、相手の哲学は相手の DB（混ぜると台帳が濁る）。"""
        philosophy, minutes = Philosophy(), Minutes()
        context = Context({"山田": "persona-yamada", "自分": "persona-運用者"})
        self_client = {"philosophy_cards_db_id": "cards-self", "persona_db_id": "persona-self"}

        result = self._run(context=context, philosophy=philosophy, minutes=minutes, self_client=self_client)

        assert result.cards == 2
        assert {call["db"] for call in philosophy.calls} == {"cards-client", "cards-self"}
        assert [group["speaker"] for group in minutes.summary] == ["山田", "自分"]

    def test_ペルソナを解決できなければ登録しない(self):
        """誤ったペルソナで書くと dedup が効かず重複を量産する（fail-close）。"""
        philosophy, minutes = Philosophy(), Minutes()

        result = self._run(context=Context({"山田": "persona-yamada"}), philosophy=philosophy, minutes=minutes)

        assert result.cards == 1 and result.cards_skipped == 1
        assert any("自分" in note and "ペルソナ" in note for note in result.notes)
        assert [call["db"] for call in philosophy.calls] == ["cards-client"]

    def test_登録に失敗したら理由を残す(self):
        result = self._run(context=Context({"山田": "p1", "自分": "p2"}),
                           philosophy=Philosophy(fail=True), minutes=Minutes())

        assert result.cards == 0 and result.cards_skipped == 2
        assert all("登録できませんでした" in note for note in result.notes)

    def test_何も出なければそう言う(self):
        result = self._run(context=Context(), philosophy=Philosophy(), minutes=Minutes(), llm=Llm("[]"))

        assert result.cards == 0 and result.notes == ["哲学カードは見つかりませんでした"]


class TestProductFacts:
    def _run(self, *, minutes, client=None, llm=None):
        return task_hub_chains.product_facts(
            Path("/tmp"), llm=llm or Llm(json.dumps([
                {"product_name": "テント", "fact_type": "spec", "key": "耐水圧", "value": "20000mm",
                 "confidence": "high"}], ensure_ascii=False)),
            prompts=PROMPTS, transcript=TRANSCRIPT, minutes_page_id="page-1",
            client=client if client is not None else {"product_facts_db_id": "facts", "product_service_db_id": "master"},
            client_name="取引先A", task_hub_minutes_lib=minutes)

    def test_抽出して登録する(self):
        minutes = Minutes()

        result = self._run(minutes=minutes)

        assert result.facts == 1 and result.facts_skipped == 1
        assert minutes.facts[0]["product_name"] == "テント"

    def test_主語と却下済みをプロンプトに入れる(self):
        """「相手が支援会社から受けているもの」を製品にしないための縛り。"""
        llm = Llm("[]")

        self._run(minutes=Minutes(rejected=["AI 記事制作"]), llm=llm)

        assert "取引先A" in llm.prompts[0] and "AI 記事制作" in llm.prompts[0]

    def test_DBが無ければ取らない(self):
        result = self._run(minutes=Minutes(), client={})

        assert result.facts == 0 and "製品 Fact の DB が無い" in result.notes[0]


def test_プロンプトが置いてある():
    assert (PROMPTS / "philosophy_cards.md").exists() and (PROMPTS / "product_facts.md").exists()


def test_抽出に失敗したら例外を上げる():
    """呼び出し側（task_hub_minutes）が受けて、理由を画面に出す。"""
    class Broken:
        def generate_text(self, prompt, **kwargs):
            raise RuntimeError("claude が失敗")

    with pytest.raises(RuntimeError, match="claude"):
        task_hub_chains.product_facts(Path("/tmp"), llm=Broken(), prompts=PROMPTS, transcript="あ",
                                    minutes_page_id="p", client={"product_facts_db_id": "f"},
                                    client_name="取引先A", task_hub_minutes_lib=Minutes())


class TestPersonaNames:
    """Notion 側のペルソナ名は会議の話者名と違うことがある（自分 → 自社自分）。"""

    def _run(self, *, context, personas):
        return task_hub_chains.philosophy(
            Path("/tmp"), llm=Llm(CARDS_JSON), prompts=PROMPTS, transcript=TRANSCRIPT,
            speakers={"山田", "自分"}, minutes_page_id="page-1", minutes_url="u",
            client={"philosophy_cards_db_id": "cards-client", "persona_db_id": "persona-client"},
            self_client={"philosophy_cards_db_id": "cards-self", "persona_db_id": "persona-self"},
            self_name="自分", context=context, philosophy_lib=Philosophy(), task_hub_minutes_lib=Minutes(),
            notion=Notion(), model="claude-opus-5", personas=personas)

    def test_対応表の名前で引く(self):
        asked = []

        class Ctx(Context):
            def resolve_persona(self, db, name):
                asked.append(name)
                return {"page_id": "p", "name": name} if name in ("自社自分", "山田") else None

        result = self._run(context=Ctx(), personas={"自分": ["自社自分"]})

        assert "自社自分" in asked and result.cards == 2

    def test_当たったペルソナ名を残す(self):
        """部分一致なので、別人に当たっていないか目で見られるようにする。"""
        class Ctx(Context):
            def resolve_persona(self, db, name):
                return {"page_id": "p", "name": "自社自分" if "自分" in name else name}

        result = self._run(context=Ctx(), personas={})

        assert any("自分 → ペルソナ「自社自分」（自社 の DB）" in note for note in result.notes)
        assert any("山田 → ペルソナ「山田」（相手 の DB）" in note for note in result.notes)
        # 所属先が引けないとき（People Master を使わない経路）は「自社／相手」で出す


class TestMultiplePersonas:
    """1 人に複数のペルソナ（連携先の設定に合わせる・運用者 指摘 2026-09-17）。"""

    class Updating(Notion):
        def __init__(self):
            self.updates = []

        def update_page_safe(self, page_id, db_id, props):
            self.updates.append((page_id, db_id, props))
            return {"ok": True}

    class Made(Philosophy):
        def register_philosophy_cards(self, cards, **kwargs):
            self.calls.append({"cards": cards, "db": kwargs["cards_db_id"], "persona": kwargs["persona_id"]})
            return {"registered": [{"title": c["title"], "id": "card-1", "url": "u"} for c in cards],
                    "skipped": 0}

    def _run(self, *, personas, context, notion, philosophy):
        return task_hub_chains.philosophy(
            Path("/tmp"), llm=Llm(CARDS_JSON), prompts=PROMPTS, transcript=TRANSCRIPT,
            speakers={"山田", "自分"}, minutes_page_id="page-1", minutes_url="u",
            client={"philosophy_cards_db_id": "cards-client", "persona_db_id": "persona-client"},
            self_client={"philosophy_cards_db_id": "cards-self", "persona_db_id": "persona-self"},
            self_name="自分", context=context, philosophy_lib=philosophy, task_hub_minutes_lib=Minutes(),
            notion=notion, model="", personas=personas)

    def test_1枚のカードに複数のペルソナを入れる(self):
        """同じ洞察を人数分コピーしない。1 枚の Persona に全員を入れる。"""
        notion, philosophy = self.Updating(), self.Made()
        context = Context({"自社社長": "p-shacho", "自社開発者": "p-dev", "山田": "p-yamada"})

        result = self._run(personas={"自分": ["自社社長", "自社開発者"]},
                           context=context, notion=notion, philosophy=philosophy)

        assert result.cards == 2                       # 山田 1 枚 ＋ 自分 1 枚（増えない）
        assert notion.updates and notion.updates[0][2]["Persona"]["relation"] == [
            {"id": "p-shacho"}, {"id": "p-dev"}]
        assert any("ペルソナ「自社社長、自社開発者」" in note for note in result.notes)

    def test_一部が見つからなくても残りで登録し名前を残す(self):
        notion, philosophy = self.Updating(), self.Made()
        context = Context({"自社社長": "p-shacho", "山田": "p-yamada"})

        result = self._run(personas={"自分": ["自社社長", "居ない人"]},
                           context=context, notion=notion, philosophy=philosophy)

        assert result.cards == 2
        assert any("「居ない人」は見つかりませんでした" in note for note in result.notes)
        assert not notion.updates                      # 1 人ぶんなので足さない


class TestPeopleMaster:
    """行き先は People Master →（relation）→ Persona の 2 段（連携先 と同じ）。"""

    class People:
        """Notion の代わり（People Master と Persona Master を持つ）。"""

        def __init__(self, rows, personas=None):
            self.rows = rows
            self.personas = personas or {}

        def query_database(self, db_id, filter=None, page_size=100):
            if filter:                                  # Persona Master（relation contains）
                person = filter["relation"]["contains"]
                return [{"id": f"persona-{index}", "properties": {"title": name}}
                        for index, name in enumerate(self.personas.get(person, []))]
            return self.rows

        @staticmethod
        def prop_multi_select(props, name):
            return props.get(name, [])

        @staticmethod
        def prop_title(props, name="title"):
            return props.get("title", "")

        @staticmethod
        def relation_val(ids):
            return {"relation": [{"id": value} for value in ids]}

        @staticmethod
        def update_page_safe(page_id, db_id, props):
            return {"ok": True}

        @staticmethod
        def llm_label(model, skill=None):
            return "label"

    FAMILY = [
        {"id": "karin", "properties": {"title": "Hanako Sato", "Speaker Labels": ["はなこ", "自分花子"],
                                       "Is Internal": {"checkbox": True}}},
        {"id": "shinpei", "properties": {"title": "Taro Sato", "Speaker Labels": ["自社自分", "自分"],
                                         "Is Internal": {"checkbox": True}}},
    ]

    def test_完全一致を先に見る(self):
        """「自分」は家族の「自分花子」にも部分一致する（2026-09-17 実データで踏んだ）。"""
        notion = self.People(self.FAMILY)

        assert task_hub_chains.find_person(notion, "people", "自分")["name"] == "Taro Sato"
        assert task_hub_chains.find_person(notion, "people", "自分花子")["name"] == "Hanako Sato"

    def test_部分一致が複数なら選ばない(self):
        rows = [{"id": "a", "properties": {"title": "Aさん", "Speaker Labels": ["田中太郎"]}},
                {"id": "b", "properties": {"title": "Bさん", "Speaker Labels": ["田中次郎"]}}]

        found = task_hub_chains.find_person(self.People(rows), "people", "田中")

        assert found["ambiguous"] == ["Aさん", "Bさん"]

    def test_その人に紐づくペルソナを全部取る(self):
        notion = self.People(self.FAMILY, personas={"shinpei": ["自社社長", "自社開発者"]})

        found = task_hub_chains.personas_of(notion, "persona-db", "shinpei")

        assert [person["name"] for person in found] == ["自社社長", "自社開発者"]

    def test_複数に当たる人のカードは登録しない(self):
        rows = [{"id": "a", "properties": {"title": "Aさん", "Speaker Labels": ["山"]}},
                {"id": "b", "properties": {"title": "Bさん", "Speaker Labels": ["山田太郎"]}}]
        notion = self.People(rows)

        result = task_hub_chains.philosophy(
            Path("/tmp"), llm=Llm(CARDS_JSON), prompts=PROMPTS, transcript=TRANSCRIPT,
            speakers={"山田"}, minutes_page_id="p", minutes_url="u",
            client={"philosophy_cards_db_id": "cards", "persona_db_id": ""},
            self_client=None, self_name="自分", context=Context(), philosophy_lib=Philosophy(),
            task_hub_minutes_lib=Minutes(), notion=notion, personas={}, people_db="people")

        assert result.cards == 0
        assert any("2 人に当たりました" in note for note in result.notes)


class TestAcrossClients:
    """商流をまたぐ人は、会議の相手ではなく**所属先**の DB へ（運用者 依頼 2026-09-17）。

    さくら歯科 ⇒ GX ⇒ 自社。GX の人の哲学は GX の DB に入れる。
    """

    class Notion(TestPeopleMaster.People):
        def __init__(self, rows, personas=None, orgs=None):
            super().__init__(rows, personas)
            self.orgs = orgs or {}

        def get_page(self, page_id):
            return {"properties": {"client_name": self.orgs.get(page_id, "")}}

        @staticmethod
        def prop_title(props, name="title"):
            return props.get(name, "") or props.get("title", "")

    ROWS = [{"id": "person_a", "properties": {
        "title": "参加者A", "Speaker Labels": ["参加者A"], "Is Internal": {"checkbox": False},
        "Organization": {"relation": [{"id": "org-gen"}]}}}]

    def test_所属先のDBに入れる(self):
        notion = self.Notion(self.ROWS, personas={"person_a": ["参加者Aさん"]}, orgs={"org-gen": "株式会社GX"})
        philosophy = TestMultiplePersonas.Made()

        class Ctx(Context):
            def resolve_client(self, name=None):
                return {"client_name": name, "philosophy_cards_db_id": f"cards-{name}",
                        "persona_db_id": f"persona-{name}"}

        result = task_hub_chains.philosophy(
            Path("/tmp"), llm=Llm(json.dumps([{"speaker": "参加者A", "title": "任せきりにしない",
                                               "content": "現場を見てから決める。" * 4,
                                               "category": "判断軸", "sensitivity": "Low", "tags": []}],
                                             ensure_ascii=False)),
            prompts=PROMPTS, transcript="参加者A: 現場を見てから決めます", speakers={"参加者A"},
            minutes_page_id="p", minutes_url="u",
            client={"client_name": "さくら歯科", "philosophy_cards_db_id": "cards-sdc",
                    "persona_db_id": "persona-sdc"},
            self_client={"client_name": "自社", "philosophy_cards_db_id": "cards-self",
                         "persona_db_id": "persona-self"},
            self_name="自分", context=Ctx(), philosophy_lib=philosophy, task_hub_minutes_lib=Minutes(),
            notion=notion, personas={}, people_db="people")

        assert result.cards == 1
        assert philosophy.calls[0]["db"] == "cards-株式会社GX"      # さくら歯科ではない
        assert any("株式会社GX の DB" in note for note in result.notes)
