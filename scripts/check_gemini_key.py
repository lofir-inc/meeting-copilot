#!/usr/bin/env python3
"""外部文字起こしの下準備が揃っているかを確かめる（音声もテキストも送らない）。

    ./scripts/check_gemini_key.py                # 確かめるだけ
    ./scripts/check_gemini_key.py --fetch-key    # キーが空なら gcloud から取り出して書く

見るのは 3 つ。キーの中身は画面にもログにも出さない。

  1. `settings.yaml` の `meeting.external_stt` が読めるか（プロジェクト・キーの場所）
  2. 課金の関門を通るか（`src/stt/external_consent.billing_enabled`）
  3. キーでモデル一覧が引けて、`gemini-3.5-transcribe` が見えるか

モデル一覧の GET だけなので、会議データの露出はゼロ。本番の音声を送る前の確認に使う。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.stt.external_consent import ExternalSttConfig, billing_enabled  # noqa: E402

MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
SERVICE = "generativelanguage.googleapis.com"


def fetch_key(project: str, key_path: Path) -> bool:
    """gcloud から API キーの文字列を取り出してファイルへ書く（0600・画面には出さない）。

    `gcloud services api-keys create` が返すのは **Operation** で、キー本体は
    `response.keyString` の下にいる。作成時に `--format="value(keyString)"` を付けても
    何も出ず、リダイレクト先が 0 バイトになる（キー自体は作られているので気づきにくい。
    2026-09-13 に実際に踏んだ）。作成とは分けて `get-key-string` で取り出す。
    """
    listed = subprocess.run(
        ["gcloud", "services", "api-keys", "list", f"--project={project}",
         "--format=value(uid,restrictions.apiTargets[0].service)"],
        capture_output=True, text=True, timeout=120,
    )
    if listed.returncode != 0:
        print(f"✗ キーを一覧できません: {listed.stderr.strip()[:160]}")
        return False
    rows = [line.split("\t") for line in listed.stdout.strip().splitlines() if line.strip()]
    matched = [row[0] for row in rows if len(row) > 1 and row[1] == SERVICE]
    if len(matched) != 1:
        print(f"✗ {SERVICE} 専用のキーが 1 つに決まりません（{len(matched)} 件）。"
              " どれを使うか決めてから書き込んでください。")
        return False

    # --project を付けないと gcloud の既定プロジェクトを見にいき、NOT_FOUND になる
    #   （uid はプロジェクトの中でしか意味を持たない。2026-09-13 に踏んだ）
    got = subprocess.run(
        ["gcloud", "services", "api-keys", "get-key-string", matched[0],
         f"--project={project}", "--format=value(keyString)"],
        capture_output=True, text=True, timeout=120,
    )
    if got.returncode != 0 or not got.stdout.strip():
        print(f"✗ キーの文字列を取り出せません: {got.stderr.strip()[:160]}")
        return False

    key_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(key_path.parent, 0o700)
    key_path.write_text(got.stdout.strip() + "\n", encoding="utf-8")
    os.chmod(key_path, 0o600)
    print(f"✓ キーを書き込みました: {key_path}（中身は出しません）")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-key", action="store_true",
                        help="キーのファイルが空なら gcloud から取り出して書き込む")
    args = parser.parse_args()

    settings = yaml.safe_load((REPO / "config" / "settings.yaml").read_text(encoding="utf-8")) or {}
    config = ExternalSttConfig.from_mapping(settings.get("meeting", {}).get("external_stt"))

    print(f"設定        : enabled={config.enabled} / project={config.billing_project or '（未設定）'}")
    if not config.billing_project:
        print("✗ 送り先のプロジェクトが未設定です（meeting.external_stt.billing_project）")
        return 1

    ok, reason = billing_enabled(config.billing_project)
    print(f"課金の関門  : {'✓' if ok else '✗'} {reason}")
    if not ok:
        return 1

    key_path = Path(config.key_file).expanduser() if config.key_file else None
    if key_path is None:
        print("✗ API キーのファイルが未設定です（meeting.external_stt.key_file）")
        return 1
    if args.fetch_key and (not key_path.exists() or not key_path.read_text(encoding="utf-8").strip()):
        if not fetch_key(config.billing_project, key_path):
            return 1
    if not key_path.exists():
        print(f"✗ API キーのファイルがありません: {key_path}（--fetch-key で取り出せます）")
        return 1
    lines = key_path.read_text(encoding="utf-8").strip().splitlines()
    key = lines[0].strip() if lines else ""
    if not key:
        # 実際に踏んだ（2026-09-13）: `gcloud services api-keys create` が返すのは **Operation** で、
        #   キー本体は `response.keyString` の下にいる。`--format="value(keyString)"` では何も出ず、
        #   リダイレクト先が 0 バイトになる（キー自体は作られているので気づきにくい）。
        #   取り出し方:
        #     gcloud services api-keys list --project=<p> --format="value(uid)"
        #     gcloud services api-keys get-key-string <uid> --format="value(keyString)" > <file>
        print(f"✗ API キーのファイルが空です: {key_path}")
        print("  キー自体は作られています。get-key-string で取り出して書き込んでください:")
        print("    gcloud services api-keys list --project="
              f"{config.billing_project} --format=\"value(uid)\"")
        return 1
    print(f"キー        : ✓ {key_path}（{len(key)} 文字。中身は出しません）")

    request = urllib.request.Request(MODELS_URL, headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read().decode("utf-8") or "{}").get("error", {})
        print(f"✗ モデル一覧を引けません: HTTP {error.code} {detail.get('status', '')} "
              f"{str(detail.get('message', ''))[:160]}")
        return 1
    except OSError as error:
        print(f"✗ 通信できません（オフライン？）: {error}")
        return 1

    names = [model["name"].split("/")[-1] for model in payload.get("models", [])]
    wanted = [name for name in names if name == config.model]
    print(f"モデル一覧  : ✓ {len(names)} 件")
    print(f"{config.model:<12}: {'✓ 見えます' if wanted else '✗ 見えません'}")
    if not wanted:
        near = [name for name in names if "transcribe" in name] or names[:5]
        print(f"  近いもの: {near}")
        return 1

    print("\n下準備は揃っています。ただし enabled は "
          f"{config.enabled} — 本番でない録音で通しを見てから true にしてください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
