#!/usr/bin/env python3
"""会議の議事録（minutes.md）を作る。使える手段を上から選び、駄目なら次へ落ちる。

    python scripts/make_minutes.py workspace/sessions/<会議>
    python scripts/make_minutes.py workspace/sessions/<会議> --engine local   # 手元の LLM で

手段の順番（`--engine auto` のとき）:
  1. Claude CLI（サブスクがあれば。従量課金なし）
  2. Gemini（API キーと課金が有効なプロジェクトがあり、**この会議の承認**が取れたとき）
  3. 手元の LLM（Ollama が動いていれば。オフラインで完結）
  4. 貼り付け用の指示書（minutes_prompt.md。何も無くても、好きな AI チャットに貼れば議事録になる）

Gemini を使うときは、外へ出す前に**その場で承認を取り、課金が有効か確かめる**（会議中と同じ関門）。
  会議の画面で既に承認していた場合は、呼び出し側が --approved-external を付ける。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.llm.claude_cli_client import ClaudeCliClient  # noqa: E402
from src.llm.gemini_client import GeminiClient, GeminiLlmConfig  # noqa: E402
from src.llm.ollama_client import LlmConfig, OllamaClient  # noqa: E402
from src import clients, known_people
from src.minutes import Availability, MinutesConfig, MinutesMaker, label, resolve_order  # noqa: E402
from src.stt.external_consent import ExternalSendRefused, ExternalSttConfig, approve, record_send  # noqa: E402


def load_config(path: Path | None) -> dict:
    for candidate in ([path] if path else []) + [REPO / "config" / "settings.yaml",
                                                  REPO / "config" / "settings.example.yaml"]:
        if candidate and candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return {}


def ask_in_terminal(session: Path, project: str) -> bool | None:
    """外へ出してよいかを端末で聞く。送るのはテキストだけなので、録音用の文面は使わない。
    聞けない（端末が無い）＝承認されていない、として None を返す。"""
    if not sys.stdin.isatty():
        return None
    answer = input(
        f"\n  議事録を作るため、会議「{session.name}」の全文と最終状態（テキストだけ・音声は送りません）を"
        f"\n  外（Google・プロジェクト {project}）へ出しますか? [y/N]: ")
    return answer.strip().lower() in {"y", "yes"}


def _client_name(session_dir) -> str:
    """その会議の相手。分からなければ空（名簿を渡さないだけで、議事録は作る）。"""
    import json

    try:
        return str(json.loads((Path(session_dir) / "client.json").read_text(encoding="utf-8")).get("name", ""))
    except (OSError, ValueError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--engine", choices=["auto", "claude_cli", "gemini", "local", "none"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--approved-external", action="store_true",
                        help="この会議で外へのテキスト送信を画面で承認済み。呼び出し側が付けるもの"
                             "（課金の確認と送信の記録はそのまま通る）")
    parser.add_argument("--no-external", action="store_true",
                        help="外（Gemini）は使わない。会議の画面で「手元で作る」を選んだ会議に付ける"
                             "（会議のあとで改めて外へ出すかを聞かない）")
    args = parser.parse_args()

    config = load_config(args.config)
    minutes_cfg = MinutesConfig.from_mapping(config.get("minutes"))
    if args.engine:
        minutes_cfg.engine = args.engine
    llm_values = dict(config.get("llm") or {})
    external_cfg = ExternalSttConfig.from_mapping((config.get("meeting") or {}).get("external_stt"))

    claude = ClaudeCliClient() if shutil.which("claude") else None

    local = None
    local_cfg = LlmConfig(**{k: v for k, v in llm_values.items() if k in LlmConfig.__dataclass_fields__})
    candidate = OllamaClient(local_cfg)
    if candidate.health_check():
        local = candidate

    # Gemini は「使える」だけでは候補にしない。承認が取れて、課金を確かめられたときだけ
    gemini = None
    wants_gemini = (not args.no_external and minutes_cfg.engine in {"auto", "gemini"} and external_cfg.enabled
                    and (minutes_cfg.engine == "gemini" or claude is None))
    if wants_gemini:
        try:
            approval = approve(external_cfg, args.session, [],
                               ask=(lambda prompt: True) if args.approved_external
                               else (lambda prompt: ask_in_terminal(args.session, external_cfg.billing_project)))
            known = set(GeminiLlmConfig.__dataclass_fields__)
            gemini = GeminiClient(GeminiLlmConfig(**{k: v for k, v in (llm_values.get("gemini") or {}).items()
                                                      if k in known}))
            record_send(args.session, approval, model=gemini.config.model, note="minutes")
        except (ExternalSendRefused, ValueError, OSError) as refused:
            print(f"  Gemini は使いません（{refused}）")
            gemini = None

    available = Availability(claude_cli=claude is not None, gemini=gemini is not None, local=local is not None)
    order = resolve_order(minutes_cfg, available)
    print(f"議事録の手段: {' → '.join(label(name) for name in order)}")

    # その会社の会議に出てくる人を集めて渡す（一覧に無い人名を「誤変換の候補」に挙げさせる）。
    #   直させるためではない。気づかせるため（`src/known_people.py` の注記）。
    client_name = _client_name(args.session)
    # People Master のその会社の人（同期のときに控えてある）。手元を読むだけ＝Notion を挟まない
    roster = clients.people_of(clients.load_cache(REPO / clients.CACHE_FILE), client_name)
    people = known_people.collect(REPO / "workspace" / "sessions", client_name, extra=roster)
    if people:
        print(f"  この会社の会議に出てくる人: {len(people)} 名"
              f"（うち People Master {len(roster)} 名）— 誤変換の候補の根拠に使います")
    maker = MinutesMaker(args.session, REPO / "prompts", minutes_cfg,
                         claude=claude, gemini=gemini, local=local, on_progress=print,
                         known_people=people)
    try:
        engine, path = maker.make(order)
    except FileNotFoundError as missing:
        print(f"✗ {missing}")
        return 1
    finally:
        if local is not None:
            local.close()
        if gemini is not None:
            gemini.close()

    # 議事録の LLM が挙げた誤変換の候補を足して、辞書の候補を作り直す（辞書に入れるのは人が選んだものだけ）
    try:
        from glossary_candidates import refresh

        candidates_path, candidates, _ = refresh(args.session, config)
        if candidates:
            print(f"  辞書の候補 {len(candidates)} 件: {candidates_path}")
    except Exception as error:  # noqa: BLE001 — 候補が出なくても議事録はできている
        print(f"  辞書の候補は作れませんでした（{error}）")

    if engine == "none":
        print(f"\n  議事録を作る手段がありませんでした。{path.name} の中身を、お使いの AI チャットに"
              f"貼り付けると議事録になります:\n    {path}")
    else:
        print(f"\n  議事録を作りました（{label(engine)}）: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
