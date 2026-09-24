#!/usr/bin/env python3
"""Common Voice（日本語）を Mozilla Data Collective から落とす。

    ./scripts/fetch_common_voice.py                 # 既定: 日本語 27.0（15.4GB）
    ./scripts/fetch_common_voice.py --slug <別の言語のスラッグ>

鍵は `~/.config/meeting-copilot/mdc.key` の 1 行目を読む（**表示もログもしない**）。
**サイトで利用条件に同意してから**でないと 403（`Terms must be accepted`）。同意は人が押す。
**urllib は Cloudflare に弾かれる**（error 1010）。curl を使う（2026-09-18 に踏んだ）。
ダウンロードの口は**スラッグではなく ID**で呼ぶ（スラッグだと `Invalid dataset ID`）。
署名つきの URL も表示しない（鍵と同じ扱い。URL を知っていれば誰でも落とせるため）。
途中で止まっても `--continue` で続きから落とす（15GB あるので、切れる前提で組む）。

落とし先は `workspace/bench/download/`（版には載らない）。展開まで済ませたら、次は:

    ./scripts/prep_common_voice.py --source workspace/bench/download/<展開先>/ja
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY_FILE = Path("~/.config/meeting-copilot/mdc.key").expanduser()
BASE = "https://mozilladatacollective.com/api"
DEFAULT_SLUG = "common-voice-scripted-speech-27-0-japane-ea5ad4e9"
OUT = REPO / "workspace" / "bench" / "download"


def read_key() -> str:
    """鍵の 1 行目。戻り値を print しない。"""
    if not KEY_FILE.is_file():
        raise SystemExit(f"鍵がありません: {KEY_FILE}\n"
                         "  サイトの Profile → Credentials で作り、1 行目に置いてください（chmod 600）")
    key = KEY_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
    if not key:
        raise SystemExit(f"鍵が空です: {KEY_FILE}")
    return key


def ask(path: str, key: str, method: str = "GET") -> dict:
    """API を叩く。鍵はコマンドラインに置かない（`ps` で見えるため）。標準入力から渡す。"""
    result = subprocess.run(
        ["curl", "-sS", "-X", method, "-w", "\n<<%{http_code}>>",
         "-H", "Accept: application/json", "-H", "Content-Type: application/json",
         "-A", "meeting-copilot/1.0 (bench)",          # UA が無いと Cloudflare が弾く（error 1010）
         "--config", "-", f"{BASE}{path}"],
        input=f'header = "Authorization: Bearer {key}"\n', text=True, capture_output=True, timeout=120)
    body, _, code = result.stdout.rpartition("\n<<")
    code = code.strip(">\n")
    if code != "200":
        raise SystemExit(f"API が {code} を返しました: {body[:200]}")
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--extract", action="store_true", help="落としたあと展開する")
    args = parser.parse_args()

    key = read_key()
    meta = ask(f"/datasets/{args.slug}", key)
    dataset_id = str(meta.get("id") or args.slug)     # download は ID で呼ぶ（スラッグだと弾かれる）
    size = int(meta.get("sizeBytes") or 0)
    name = str(meta.get("filename") or f"{args.slug}.tar.gz")
    print(f"{meta.get('name')}／{size / 1073741824:.1f} GB／{meta.get('license')}")

    made = ask(f"/datasets/{dataset_id}/download", key, method="POST")
    url = made.get("url") or made.get("downloadUrl") or made.get("signedUrl")
    if not url:
        raise SystemExit(f"ダウンロードの口が返りませんでした（返ってきた項目: {sorted(made)}）")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / name
    print(f"落とし先: {target}\n（署名つき URL は表示しません。途中で切れたら同じコマンドで続きから）\n")
    # URL をコマンドラインに置くと `ps` で見えるので、標準入力から渡す
    result = subprocess.run(
        ["curl", "--location", "--continue-at", "-", "--fail", "--progress-bar",
         "--output", str(target), "--config", "-"],
        input=f'url = "{url}"\n', text=True)
    if result.returncode != 0:
        raise SystemExit(f"落としきれませんでした（curl {result.returncode}）。同じコマンドで続きから再開できます")

    got = target.stat().st_size
    print(f"\n落ちました: {got / 1073741824:.1f} GB")
    if size and got < size:
        print(f"まだ途中です（{got / size:.0%}）。もう一度実行すると続きから落とします")
        return 1
    if args.extract:
        print("展開しています（15GB ぶんなので数分）…")
        subprocess.run(["tar", "-xzf", str(target), "-C", str(args.out)], check=False)
        print("展開しました:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
