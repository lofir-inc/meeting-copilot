"""ローリング会議状態のテスト。"""

from src.llm.meeting_state import Item, MeetingState, StateDelta, apply_delta


def delta(**values: object) -> StateDelta:
    """テスト用の最小差分を作る。"""
    base = {"summary": "要約", "current_topic": "", "topic_changed": False, "new_decisions": [], "new_todos": [], "new_questions": [], "updates": [], "next_asks": []}
    base.update(values)
    return StateDelta(**base)


def test_apply_delta_assigns_ids_and_deduplicates() -> None:
    """項目には種別ごとの連番を振り、似た項目は追加しない。"""
    state, changes = apply_delta(MeetingState(), delta(new_decisions=[{"text": "A を採用する", "by": "田中"}], new_todos=[{"text": "資料を作る", "owner": "田中", "due": "未定"}]), 10.0)
    state, next_changes = apply_delta(state, delta(new_decisions=[{"text": "Ａを採用する。", "by": "田中"}], new_todos=[{"text": "資料を 作る。", "owner": "田中", "due": "未定"}]), 20.0)
    assert [item.id for item in state.decisions] == ["D1"]
    assert [item.id for item in state.todos] == ["T1"]
    assert len(changes) == 2
    assert next_changes == []


def test_apply_delta_keeps_ids_and_reports_unknown_update() -> None:
    """未知 ID は警告だけにして既存項目は消さない。"""
    previous = MeetingState(decisions=[Item("D7", "既存決定")])
    state, changes = apply_delta(previous, delta(updates=[{"id": "D9", "status": "done", "note": "なし"}]), 30.0)
    assert [item.id for item in state.decisions] == ["D7"]
    assert changes == ["⚠ 未知の id: D9"]


def test_summary_is_limited_to_300_characters() -> None:
    """LLM の長すぎる要約を保存しない。"""
    state, _ = apply_delta(MeetingState(), delta(summary="あ" * 401), 1.0)
    assert len(state.summary) == 300


def test_apply_delta_keeps_previous_summary_and_topic_for_empty_values() -> None:
    """空の要約と論点は前回の状態を保持する。"""
    previous = MeetingState(summary="前の要約", current_topic="前の論点")
    state, _ = apply_delta(previous, delta(summary="  ", current_topic="\t"), 1.0)
    assert state.summary == "前の要約"
    assert state.current_topic == "前の論点"


def test_prompt_compacts_old_open_decisions() -> None:
    """開いた決定は新しい八件だけを載せて古い決定を畳む。"""
    state = MeetingState(decisions=[Item(f"D{index}", f"決定 {index}", created_at=float(index)) for index in range(1, 11)])
    prompt = state.to_prompt_text()
    assert "他 2 件の決定（state.json 参照）" in prompt
    assert "[D1]" not in prompt
    assert "[D2]" not in prompt
    assert "[D10]" in prompt


def test_prompt_limit_preserves_decisions_and_todos_before_topics_questions() -> None:
    """プロンプト短縮時に決定と TODO を先に落とさない。"""
    state = MeetingState(
        updated_at=1000.0,
        summary="要約" * 100,
        topics=[Item("P1", "論点" * 30), Item("P2", "古い論点" * 30)],
        decisions=[Item("D1", "決定事項")],
        todos=[Item("T1", "TODO事項", by="田中", due="明日")],
        questions=[Item("Q1", "質問" * 30)],
    )
    prompt = state.to_prompt_text(180)
    assert len(prompt) <= 180
    assert "D1" in prompt
    assert "T1" in prompt
    assert "P1" not in prompt


def test_from_json_roundtrip() -> None:
    """JSON 往復で状態を失わない。"""
    source = MeetingState(decisions=[Item("D1", "決定", by="佐藤")], next_asks=["確認する"])
    assert MeetingState.from_json(source.to_json()).to_dict() == source.to_dict()


def test_to_markdown_has_main_sections() -> None:
    """議事録には論点の流れを含む主要区画を出す。"""
    markdown = MeetingState().to_markdown("テスト会議")
    assert "## 論点の流れ" in markdown
    assert "## 決定事項" in markdown
    assert "## TODO" in markdown
    assert "## 未解決の質問" in markdown
    assert "## 次に聞くべきこと" in markdown


def test_prompt_text_caps_open_todos_and_truncates_long_text():
    """08-26 で TODO 69 件・13,551 字に膨らんだ回帰。open は新しい 20 件＋「他 N 件」。"""
    state = MeetingState(updated_at=1000.0)
    for index in range(25):
        state.todos.append(Item(id=f"T{index + 1}", text="長い作業 " * 30, status="open", by="自分", due="未定", created_at=float(index), updated_at=float(index), note="注記 " * 40))
    text = state.to_prompt_text(max_chars=100_000)
    assert "- 他 5 件のTODO（state.json 参照）" in text
    assert "[T25]" in text and "[T1]" not in text
    for line in text.splitlines():
        assert len(line) < 220
    # max_chars に収めるときも「他 N 件」行は残り、古い open から落ちる
    short = state.to_prompt_text(max_chars=1200)
    assert len(short) <= 1200
    assert "他 " in short and "[T25]" in short


def test_topic_change_creates_timeline_and_closes_previous_topic() -> None:
    """論点が変わると時系列項目を追加して直前を閉じる。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算"), 10.0)
    state, _ = apply_delta(state, delta(current_topic="採用", topic_changed=True), 120.0)
    assert [(item.id, item.status, item.updated_at) for item in state.topics] == [("P1", "closed", 120.0), ("P2", "active", 120.0)]


def test_topic_does_not_change_without_llm_flag_or_before_floor() -> None:
    """LLM が topic_changed を立てないか、床（90 秒）の前なら論点は切らない。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算"), 10.0)
    state, changes = apply_delta(state, delta(current_topic="採用スケジュール"), 120.0)
    assert [item.id for item in state.topics] == ["P1"] and changes == []
    state, changes = apply_delta(state, delta(current_topic="採用スケジュール", topic_changed=True), 60.0)
    assert [item.id for item in state.topics] == ["P1"] and changes == []
    assert state.current_topic == "採用スケジュール"


def test_same_topic_does_not_add_timeline_item() -> None:
    """正規化後に同じ論点は時系列項目を増やさない。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算案"), 10.0)
    state, _ = apply_delta(state, delta(current_topic="予 算案。"), 20.0)
    assert [item.id for item in state.topics] == ["P1"]


def test_asks_are_numbered_and_deduplicated() -> None:
    """次に聞くことは A 番号を持ち、開いた問いと重複しない。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["上限額は"]), 10.0)
    state, _ = apply_delta(state, delta(next_asks=["上限額は？"]), 20.0)
    assert [(item.id, item.by) for item in state.asks] == [("A1", "P1")]


def test_asked_update_records_answer_note() -> None:
    """聞いて回答を得た次に聞くことは asked と回答要旨を記録する。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["上限額は"]), 10.0)
    state, _ = apply_delta(state, delta(updates=[{"id": "A1", "status": "asked", "note": "100万円"}]), 20.0)
    assert (state.asks[0].status, state.asks[0].note) == ("asked", "100万円")


def test_topic_change_keeps_open_asks_until_finalize() -> None:
    """論点が移っても問いは open のまま残り、finalize で skipped に確定する。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["上限額は"]), 10.0)
    state, _ = apply_delta(state, delta(current_topic="採用", topic_changed=True), 120.0)
    assert state.asks[0].status == "open" and state.asks[0].by == "P1"
    assert "- △ [A1] 上限額は" in state.to_markdown("t")
    changes = state.finalize(200.0)
    assert changes == ["~ [A1] 上限額は（open → skipped）"]
    assert state.asks[0].status == "skipped" and state.next_asks == []
    assert state.finalize(201.0) == []


def test_llm_update_cannot_close_topic_and_same_topic_is_not_duplicated() -> None:
    """LLM が updates で P を閉じても無視し、同文の論点を二重に立てない。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="議論内容の確認"), 10.0)
    state, changes = apply_delta(state, delta(updates=[{"id": "P1", "status": "superseded", "note": ""}]), 60.0)
    assert state.topics[0].status == "active" and changes == ["⚠ 未知の id: P1"]
    state, changes = apply_delta(state, delta(current_topic="議論内容の確認", topic_changed=True), 150.0)
    assert [item.id for item in state.topics] == ["P1"] and changes == []


def test_next_asks_syncs_to_open_asks() -> None:
    """互換用 next_asks は新しい open の asks だけに同期する。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["質問1", "質問2"]), 10.0)
    state, _ = apply_delta(state, delta(updates=[{"id": "A2", "status": "asked", "note": "回答"}]), 20.0)
    assert state.next_asks == ["質問1"]


def test_from_json_reads_legacy_state_without_asks_or_topics() -> None:
    """asks と topics が無い旧形式の state.json を読める。"""
    state = MeetingState.from_json('{"current_topic":"旧論点","next_asks":["確認"]}')
    assert state.current_topic == "旧論点"
    assert state.topics == []
    assert state.asks == []


def test_markdown_shows_topic_flow_and_ask_marks() -> None:
    """議事録は論点ごとの次に聞くことの結果を記号で表示する。"""
    state = MeetingState(
        topics=[Item("P1", "予算", "active", created_at=10.0)],
        asks=[Item("A1", "質問1", "open", by="P1"), Item("A2", "質問2", "asked", by="P1", note="回答"), Item("A3", "質問3", "skipped", by="P1")],
    )
    markdown = state.to_markdown("テスト会議")
    assert "## 論点の流れ" in markdown
    assert "○ [A1]" in markdown and "✓ [A2]" in markdown and "— [A3]" in markdown


def test_reworded_topic_does_not_create_new_timeline_item() -> None:
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算の配分", next_asks=["来期の上限は？"]), 30.0)
    state, changes = apply_delta(state, delta(current_topic="予算配分について", topic_changed=True), 150.0)
    assert [item.id for item in state.topics] == ["P1"]
    assert state.asks[0].status == "open"
    assert changes == []


def test_echoed_ask_lines_are_cleaned_and_not_duplicated() -> None:
    """LLM が状態の行を写して返しても、id タグと状態を落として既存と突き合わせる。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["来期の上限は？"], new_questions=[{"text": "納期はいつか", "asked_by": "田中"}]), 30.0)
    state, changes = apply_delta(state, delta(next_asks=["[A1] 来期の上限は？（open）", "[Q1] 納期はいつか", "[A1] [A2] 来期の上限は？ / 担当・発言者: P1", "承認者は誰か"]), 60.0)
    assert [item.text for item in state.asks] == ["来期の上限は？", "承認者は誰か"]
    assert changes == ["+ [A2] 承認者は誰か（open）"]
    assert "次に聞くこと（未）:\n- [A2] 承認者は誰か\n- [A1] 来期の上限は？" in state.to_prompt_text()
    assert "担当・発言者: P1" not in state.to_prompt_text()


def test_open_asks_are_capped_per_topic() -> None:
    """同じ論点の open な問いは 3 件まで。論点が変われば新しい枠になる。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["問い1", "問い2", "問い3"]), 10.0)
    state, changes = apply_delta(state, delta(next_asks=["問い4", "問い5"]), 40.0)
    assert [item.id for item in state.asks] == ["A1", "A2", "A3"] and changes == []
    state, _ = apply_delta(state, delta(updates=[{"id": "A1", "status": "asked", "note": "回答"}], next_asks=["問い4"]), 70.0)
    assert [item.text for item in state.asks if item.status == "open"] == ["問い2", "問い3", "問い4"]
    state, _ = apply_delta(state, delta(current_topic="採用", topic_changed=True, next_asks=["問い6"]), 200.0)
    assert [(item.text, item.by) for item in state.asks if item.status == "open"] == [("問い2", "P1"), ("問い3", "P1"), ("問い4", "P1"), ("問い6", "P2")]


def test_ask_status_from_llm_is_normalized() -> None:
    """問いの状態は open / asked / skipped に正規化する。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["上限は", "承認者は"]), 10.0)
    state, _ = apply_delta(state, delta(updates=[{"id": "A1", "status": "answered", "note": "上限は（500万）"}, {"id": "A2", "status": "superseded", "note": "トピックが変更されたため一旦保留"}]), 40.0)
    assert [(item.id, item.status) for item in state.asks] == [("A1", "asked"), ("A2", "open")]
    assert state.asks[0].note == "500万"


def test_resuggested_ask_is_not_duplicated_regardless_of_status() -> None:
    """同じ問いを LLM が再提案しても、状態を問わず二重に登録しない。"""
    state, _ = apply_delta(MeetingState(), delta(current_topic="予算", next_asks=["来期の上限は？"]), 30.0)
    state, _ = apply_delta(state, delta(current_topic="採用スケジュール", topic_changed=True), 150.0)
    state.finalize(150.0)
    assert state.asks[0].status == "skipped"
    state, changes = apply_delta(state, delta(next_asks=["来期の上限は"]), 180.0)
    assert [(item.id, item.status) for item in state.asks] == [("A1", "skipped")]
    assert changes == []
