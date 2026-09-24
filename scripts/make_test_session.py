#!/usr/bin/env python3
"""外部文字起こしの経路を通しで見るための、**合成音声のセッション**を作る。

    ./scripts/make_test_session.py            # workspace/sessions/selftest-external を作る

なぜ合成音声か: 通しを見るには外へ音声を出すことになる。手元に残っている録音は
どれも**実在の相手の声**（クライアント・インタビュー相手）なので、試験には使えない。
macOS の `say` で作った音声なら、誰の声でもなく、話された内容も全部こちらで決められる。

作るもの（本番と同じ 2 系統）:
  recording_remote.wav … 相手側。2 人ぶんを別の声で入れる（話者の割り当てを試すため）
  recording_self.wav   … 自分側。自分の位置

台本には、辞書で直るはず語（クラウド→Claude）と、直ってはいけない語
（クラウドファンディング）と、相づちを入れてある。通しで印と置き換えを一度に見られる。

**これは経路を見るためのもので、話者判定の精度を測るものではない。** 30 秒台では
不明話者プールが育ちきらず（`min_unknown_sec` 6 秒・`min_pool_sec` 3 秒）、同じ人に
複数のラベルが付く。精度は実会議で測る（2026-09-11 の 50 分で 97%）。
ここで見るのは「2 つの声が別物として扱われているか」まで。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 48000

# (声, 台本)。声を分けるのは、手元の話者割り当てが効くかを見るため。
#   使う声は `say -v '?'` に出るものだけ。入っていない声を指定しても say は**エラーを返さず**
#     既定の声で喋る（終了コード 0）。2026-09-13 に踏んだ: Otoya / Alex が入っておらず、
#     相手側の 2 人が同じ声で作られていて、話者が分かれないのを実装のせいだと誤解しかけた。
REMOTE = [
    ("Kyoko", "おはようございます。今日はよろしくお願いします。"),
    ("Grandpa", "はい。うん。うんうん。"),
    ("Kyoko", "先日の件ですが、クラウドで文字起こしを試してみました。"),
    ("Grandpa", "なるほど。クラウドファンディングの案件はどうなりましたか。"),
    ("Kyoko", "そちらは来月からです。ワードプレスの改修も合わせて進めます。"),
    ("Grandpa", "了解です。うん。はい。"),
]
SELF = [
    ("Reed", "お世話になります。自社の自分です。"),
    ("Reed", "はい。うんうん。"),
    ("Reed", "クラウドコードで直すところまで見ておきます。"),
    ("Reed", "ありがとうございます。ではそのように進めましょう。"),
]

GAP_SEC = 1.5
"""台詞のあいだに入れる無音。

0.6 秒だと短すぎた（2026-09-13 の通し）: `gemini_transcribe.py` は語の間が 0.8 秒以内なら
同じ発話としてつなぐので、24 秒ぜんぶが **1 発話** になり、話者の割り当ても相づちの印も
試せなかった。実際の会議はもっと間が空く（09-11 の 50 分で 565 区間）。
"""


def installed_voices() -> set[str]:
    """`say -v '?'` に出る声の名前。"""
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    return {line.split()[0] for line in result.stdout.splitlines() if line.split()}


def check_voices(lines: list[tuple[str, str]]) -> None:
    """入っていない声を使おうとしていたら止める。

    `say` は知らない声を渡しても**エラーにならず**既定の声で喋る（終了コード 0）。
    黙って別の声になるので、話者を分ける試験が意味を失う（2026-09-13 に踏んだ）。
    """
    available = installed_voices()
    missing = sorted({voice for voice, _ in lines} - available)
    if missing:
        raise SystemExit(
            f"この Mac に入っていない声です: {missing}\n"
            f"  使えるのは: {sorted(available)[:20]} …\n"
            "  say は知らない声でもエラーを出さず既定の声で喋るので、ここで止めます。"
        )


def synthesize(lines: list[tuple[str, str]], out_path: Path, tmp: Path) -> float:
    """台本を 1 本の wav につなぐ。戻り値は長さ（秒）。"""
    pieces: list[np.ndarray] = []
    for index, (voice, text) in enumerate(lines):
        chunk_path = tmp / f"{out_path.stem}-{index}.wav"
        result = subprocess.run(
            ["say", "-v", voice, "-o", str(chunk_path),
             "--data-format=LEI16@48000", text],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"say に失敗しました（声 {voice}）: {result.stderr.strip()[:160]}")
        audio, rate = sf.read(str(chunk_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"想定外のサンプリングレート: {rate}")
        pieces.append(audio)
        pieces.append(np.zeros(int(GAP_SEC * SAMPLE_RATE), dtype=np.float32))
        chunk_path.unlink(missing_ok=True)
    joined = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    sf.write(str(out_path), joined, SAMPLE_RATE, subtype="PCM_16")
    return joined.size / SAMPLE_RATE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="selftest-external")
    args = parser.parse_args()

    session = REPO / "workspace" / "sessions" / args.name
    session.mkdir(parents=True, exist_ok=True)
    tmp = session / "_tmp"
    tmp.mkdir(exist_ok=True)

    check_voices(REMOTE + SELF)
    remote_sec = synthesize(REMOTE, session / "recording_remote.wav", tmp)
    self_sec = synthesize(SELF, session / "recording_self.wav", tmp)
    tmp.rmdir()

    # 会議中の訂正は無し、要約も無しで始める（作り直しの経路だけを見る）
    (session / "corrections.jsonl").write_text("", encoding="utf-8")

    print(f"作りました: {session}")
    print(f"  recording_remote.wav  {remote_sec:.1f} 秒（2 人ぶんを別の声で）")
    print(f"  recording_self.wav    {self_sec:.1f} 秒")
    print("\n通しで見るには:")
    print(f"  ./scripts/finalize_meeting.py workspace/sessions/{args.name} --external-stt")
    print("\n合成音声なので、外へ出しても誰の声でもない。台本には辞書で直る語（クラウド→Claude）と、")
    print("  直ってはいけない語（クラウドファンディング）と、相づちが入れてある。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
