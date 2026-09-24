#!/usr/bin/env python3
"""この Mac で動くか、足りないものは何かを 1 枚で出す。

    python scripts/check_prereqs.py            # 見るだけ
    python scripts/check_prereqs.py --quiet    # 足りないものだけ

まっさらな Mac に入れたとき、最初に困るのは「何が足りないのか分からない」こと。
  足りないものは**動かしてみて初めて**分かり、しかも出るのは
  `[Errno 2] No such file or directory: 'ffmpeg'` のような形になる。

  ここでは「**何が無いと、何ができないか**」を先に出す。
  足りなくても会議そのものは回ることが多い（手元の文字起こしはモデルさえあれば動く）。
  必須と任意を分けて出す。
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (表示名, import する名前, 何に使うか)
PACKAGES = [
    ("numpy", "numpy", "音声の計算"),
    ("sounddevice", "sounddevice", "マイクからの取り込み"),
    ("mlx-whisper", "mlx_whisper", "手元の文字起こし"),
    ("resemblyzer", "resemblyzer", "話者の割り当て（声紋）"),
    ("silero-vad", "silero_vad", "話の切れ目の判定"),
    ("onnxruntime", "onnxruntime", "同上（silero が読み込む）"),
    ("fastapi", "fastapi", "会議中の画面"),
    ("uvicorn", "uvicorn", "同上"),
    ("pyyaml", "yaml", "設定の読み込み"),
    ("httpx", "httpx", "外のエンジンとのやり取り"),
    ("websockets", "websockets", "同上（会議中に送る場合）"),
    ("ScreenCaptureKit", "ScreenCaptureKit", "相手の声の取り込み（システム音声）"),
]


class Result:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.missing_required: list[str] = []
        self.missing_optional: list[str] = []

    def heading(self, text: str) -> None:
        if not self.quiet:
            print(f"\n── {text} ──")

    def show(self, ok: bool, name: str, note: str, *, required: bool) -> None:
        mark = "✅" if ok else ("✗" if required else "—")
        if not (ok and self.quiet):          # --quiet では、足りないものだけ出す
            print(f"  {mark} {name}{'  ' + note if note else ''}")
        if not ok:
            (self.missing_required if required else self.missing_optional).append(name)


def macos_version() -> tuple[int, int]:
    text = platform.mac_ver()[0] or "0.0"
    parts = [int(piece) for piece in text.split(".")[:2]] + [0]
    return parts[0], parts[1]


def ollama_models() -> list[str]:
    if not shutil.which("ollama"):
        return []
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()]


def settings_model() -> str:
    """設定が要求している要約のモデル名（見本を含めて探す）。

    `pip install` の**前**にも動かせるように、pyyaml が無くても落ちない。
    """
    try:
        import yaml  # noqa: PLC0415 — まだ入っていないことがある
    except ImportError:
        return ""
    for name in ("settings.yaml", "settings.example.yaml"):
        path = REPO / "config" / name
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return str((data.get("llm") or {}).get("model", ""))
    return ""


def compiler() -> bool:
    """C コンパイラ（Xcode Command Line Tools）があるか。

    なぜ要るか（2026-09-20 に実測）: 依存のうち **`webrtcvad`（resemblyzer が使う）だけ
      ホイールが無く、その場でソースからコンパイルされる**。まっさらな Mac で
      Command Line Tools が入っていないと、`pip install -r requirements.txt` が
      コンパイルエラーで止まる。Homebrew を入れると一緒に入るので、普通は気づかない。
    """
    if shutil.which("cc") is None:
        return False
    return subprocess.run(["xcode-select", "-p"], capture_output=True).returncode == 0


def whisper_cached() -> bool:
    """文字起こしのモデルが取得済みか（初回は約 1.5 GB を取りにいく）。"""
    home = Path.home() / ".cache" / "huggingface" / "hub"
    return home.exists() and any(home.glob("models--*whisper*"))


def screen_permission() -> bool | None:
    """このアプリ（ターミナル）に画面収録の許可があるか。分からなければ None。"""
    try:
        from Quartz import CGPreflightScreenCaptureAccess  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — pyobjc が無い環境
        return None
    try:
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:  # noqa: BLE001
        return None


def free_gb() -> float:
    usage = shutil.disk_usage(REPO)
    return usage.free / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="足りないものだけ出す")
    args = parser.parse_args()
    result = Result(quiet=args.quiet)

    result.heading("この Mac")
    major, minor = macos_version()
    result.show((major, minor) >= (12, 3), "macOS 12.3 以上",
                f"（いま {platform.mac_ver()[0] or '不明'}）", required=True)
    arm = platform.machine() == "arm64"
    result.show(arm, "Apple Silicon",
                f"（いま {platform.machine()}）"
                + ("" if arm else " … 手元の文字起こしは動きません。外のエンジンを使ってください"),
                required=False)
    result.show(sys.version_info >= (3, 10), "Python 3.10 以上",
                f"（いま {platform.python_version()}・{sys.executable}）", required=True)
    result.show(free_gb() >= 5, "空き容量 5 GB 以上", f"（いま {free_gb():.0f} GB）", required=True)

    print("\n── 依存（pip install -r requirements.txt）──")
    for name, module, use in PACKAGES:
        try:
            importlib.import_module(module)
            ok = True
        except Exception:  # noqa: BLE001 — 読み込めない＝使えない
            ok = False
        result.show(ok, name, f"… {use}", required=True)

    result.heading("外の道具")
    result.show(bool(shutil.which("ffmpeg")), "ffmpeg",
                "… 会議のあとの作り直しと録音の圧縮（brew install ffmpeg）", required=True)
    result.show(bool(shutil.which("ffprobe")), "ffprobe", "… 同上（ffmpeg に同梱）", required=True)

    model = settings_model()
    models = ollama_models()
    has_model = any(name.split(":")[0] == model.split(":")[0] for name in models) if model else False
    result.show(bool(shutil.which("ollama")), "ollama",
                "… 手元で要約・議事録（無くても会議は回ります）", required=False)
    if shutil.which("ollama"):
        result.show(has_model, f"ollama のモデル {model}",
                    f"（入っているもの: {'／'.join(models) or 'なし'}）", required=False)
    result.show(bool(shutil.which("claude")), "claude（Claude Code）",
                "… 議事録と裏取りをいちばん良い手段で（サブスク）", required=False)
    result.show(bool(shutil.which("gcloud")), "gcloud",
                "… Gemini を使うときだけ（課金の確認に使います）", required=False)

    result.heading("取得済みのもの")
    result.show(whisper_cached(), "文字起こしのモデル",
                "… 初回の起動で約 1.5 GB を取りにいきます（ネットが要ります）", required=False)

    result.heading("macOS の許可")
    screen = screen_permission()
    if screen is None:
        if not result.quiet:
            print("  ? 画面収録  … 判定できませんでした（システム設定で確かめてください）")
    else:
        result.show(screen, "画面収録",
                    "… これが無いと**相手の声が完全に無音**になります（SETUP.md 3）", required=True)
    if not result.quiet:
        print("  ? マイク  … ここからは判定できません。一度起動して「測る（3秒）」で確かめてください")

    result.heading("まとめ")
    if result.missing_required:
        print(f"  ✗ 足りないもの（これが無いと動きません）: {'／'.join(result.missing_required)}")
    else:
        print("  ✅ 必須のものは揃っています")
    if result.missing_optional:
        print(f"  — 無くても動きますが、できないことがあります: {'／'.join(result.missing_optional)}")
    print()
    return 1 if result.missing_required else 0


if __name__ == "__main__":
    sys.exit(main())
