#!/usr/bin/env python3
"""会議のあとに、録音から全文を起こし直して議事録の材料を作り直す。

会議中はリアルタイムに追いつく必要があるので、取りこぼしや遅れが出る（2026-09-11 の本番は 0.7 倍速で
17 分遅れた）。会議が終われば急ぐ理由はなく、同じ録音を 12 倍速で流し直せる（49 分の会議で約 4 分）。
会議中の画面は「その場で見るため」、議事録は「あとで作り直したもの」と役割を分ける。

    python scripts/finalize_meeting.py workspace/sessions/2026-09-11-2

やること:
  1. recording_remote.wav を本番と同じ VAD → 話者判定 → 文字起こしに通す（speakers.json を引き継ぐ）
  2. recording_self.wav を自分の発言として文字起こしする（話者判定はしない）
  3. 会議中に画面で直した話者（corrections.jsonl）を反映する
  4. transcripts_final.jsonl と minutes_input.md を書き直す（元のファイルは .live. に退避）

会議中に外（Gemini）で起こしていたとき（`stt.engine: gemini`＝方式④）は `--reuse-live` で、
  **同じ音声を送り直さない**。会議中の全文をそのまま使い、
    ・話者だけを会議が終わったあとの声紋で当て直す（会議中に付けた名前が効く）
    ・外へ送れなかった区間（`external_gaps.jsonl`）だけを拾い直す
  ＝ 費用は会議中のぶんだけで済み、精度は落ちない。引数を省くと、送信の記録を見て自動で決める。

要約（state.json）は作り直さない。必要なら scripts/replay_state.py を続けて回す。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replay_meeting import decode_audio, replay  # noqa: E402

from src.audio import ffmpeg  # noqa: E402
from src.audio.vad import ChunkBuilder, VadConfig  # noqa: E402
from src.audio.turn_splitter import TurnSplitConfig  # noqa: E402
from src.dispatch import write_minutes_handoff  # noqa: E402
from src.enrollment import EnrollmentConfig, failed_enrollment_lines, load_or_create  # noqa: E402
from src.audio.enrolled_diarizer import DiarizerConfig, EnrolledDiarizer  # noqa: E402
from src.llm.meeting_state import MeetingState  # noqa: E402
from src.stt.whisper_client import SttConfig, TranscriptSegment, WhisperClient  # noqa: E402
from src.text.glossary import apply_glossary, load_glossary, load_protected  # noqa: E402
from src.text.terms import load_terms  # noqa: E402
from src.text import task_hub_dictionary  # noqa: E402
from src.text.aizuchi import mark_rows  # noqa: E402
from src.audio.segment_labeler import label_segments  # noqa: E402
from src.stt.gemini_transcribe import TranscribeApi, TranscribeConfig, load_key  # noqa: E402
from src.meeting_orchestrator import RENAMES_FILE  # noqa: E402
from src.stt.live_batch import GAPS_FILE  # noqa: E402
from src.stt.external_consent import (  # noqa: E402
    ExternalSendRefused,
    ExternalSttConfig,
    approve,
    record_send,
)

REPO = Path(__file__).resolve().parent.parent


def transcribe_self(audio, sample_rate: int, vad_cfg: VadConfig, whisper: WhisperClient, speaker: str, block_sec: float) -> list[dict]:
    """自分側の録音を、話者判定なしで文字起こしする。"""
    builder = ChunkBuilder(speaker, vad_cfg, sample_rate)
    block = int(block_sec * sample_rate)
    chunks = []
    for index in range((audio.size + block - 1) // block):
        frame = audio[index * block : (index + 1) * block]
        if frame.size:
            chunks.extend(builder.feed(frame))
    chunks.extend(builder.feed(np.zeros(int(vad_cfg.silence_duration_sec * sample_rate) + block, dtype=np.float32)))
    rows = []
    for chunk in chunks:
        segment = whisper.transcribe(chunk)
        if segment is not None:
            rows.append({"speaker": speaker, "text": segment.text,
                         "start_time": round(chunk.start_time, 2), "end_time": round(chunk.end_time, 2)})
    return rows


def apply_corrections(rows: list[dict], corrections: list[dict], tolerance: float = 1.0) -> int:
    """会議中に画面で直した話者を、いちばん時刻の近い行へ反映する。"""
    applied = 0
    for correction in corrections:
        start = float(correction.get("start_time", -1))
        new = str(correction.get("new", "")).strip()
        if not new:
            continue
        near = [row for row in rows if abs(row["start_time"] - start) <= tolerance]
        if not near:
            continue
        target = min(near, key=lambda row: abs(row["start_time"] - start))
        if target["speaker"] != new:
            target["speaker"] = new
            applied += 1
    return applied


def transcribe_external(
    session: Path,
    remote_path: Path,
    self_path: Path,
    self_name: str,
    cfg: ExternalSttConfig,
    sample_rate: int,
    diarizer,
    approved: bool = False,
) -> list[dict]:
    """外（Gemini）に文字起こしさせ、話者の割り当ては手元で行う。

    関門（`src/stt/external_consent.py`）を通らなければ、ここへは来ない。
    `scripts/gemini_transcribe.py` の側でも課金をもう一度確かめる（二重の fail-closed）。
    """
    files = [path for path in (remote_path, self_path) if path.exists()]
    # `approved` は、呼び出し側（会議の画面）で人が答えた直後にだけ立つ。
    #   立っていても課金の確認と送信の記録はそのまま通る（関門を飛ばすのは「聞く」ところだけ）。
    approval = approve(cfg, session, files, ask=(lambda prompt: True) if approved else None)
    record_send(session, approval, model=cfg.model, note="finalize_meeting")

    script = REPO / "scripts" / "gemini_transcribe.py"

    def run(audio: Path, out: Path, speaker: str | None) -> list[dict]:
        argv = [sys.executable, str(script), str(audio), "--out", str(out),
                "--model", cfg.model, "--billing-project", cfg.billing_project]
        if cfg.key_file:
            argv += ["--key-file", cfg.key_file]
        if speaker:
            argv += ["--speaker", speaker]
        print(f"  外へ送っています: {audio.name}")
        completed = subprocess.run(argv, check=False, timeout=cfg.timeout_sec)
        if completed.returncode != 0:
            raise ExternalSendRefused(f"{audio.name} の文字起こしに失敗しました")
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    remote_rows = run(remote_path, session / "gemini_remote.jsonl", None)
    remote_audio = decode_audio(remote_path, sample_rate)
    counts = label_segments(remote_rows, remote_audio, sample_rate, diarizer)
    print(f"  話者は手元で当てました: {counts}")

    self_rows: list[dict] = []
    if self_path.exists():
        self_rows = run(self_path, session / "gemini_self.jsonl", self_name)

    rows = [{"speaker": row["speaker"], "text": row["text"],
             "start_time": row["start_time"], "end_time": row["end_time"]}
            for row in remote_rows + self_rows if row.get("text")]
    return sorted(rows, key=lambda row: row["start_time"])


# ---------------------------------------------------------------------------
# 方式④（会議中に外で起こした）の作り直し — 送り直さず、話者だけ当て直す
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    """壊れた行は読み飛ばして読む。"""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sent_live(session: Path) -> bool:
    """この会議は、会議**中**に外へ送っていたか（送信の記録で判る）。"""
    return any(entry.get("note") == "live_batch"
               for entry in read_jsonl(session / "external_sends.jsonl"))


def apply_renames(rows: list[dict], renames: list[dict]) -> int:
    """画面で付けた話者名（旧名 → 新名）を、当て直したあとの行へ反映する。

    声紋に結び付いた名前なら当て直しで再現されるが、**結び付かないまま付けた名前**は
    作り直すと消える（2026-09-14 の会議で 736 行が「不明話者1」へ戻った）。記録から当て直す。
    """
    mapping: dict[str, str] = {}
    for entry in renames:
        old_name, new_name = str(entry.get("old", "")), str(entry.get("new", ""))
        if not old_name or not new_name:
            continue
        # 途中で二重に改名していたら、最後の名前まで辿る
        mapping = {key: (new_name if value == old_name else value) for key, value in mapping.items()}
        mapping[old_name] = new_name
    applied = 0
    for row in rows:
        target = mapping.get(row.get("speaker", ""))
        if target:
            row["speaker"] = target
            applied += 1
    return applied


def relabel_live_rows(rows: list[dict], remote_audio, sample_rate: int, diarizer, self_name: str) -> dict[str, int]:
    """会議中に外で起こした行の話者を、会議が終わったあとの声紋で当て直す。

    会議中は声紋が育っていない（冒頭ほど弱い）。終わってからなら、画面で付けた名前も
    増えた発話も全部使えるので、同じ行でも当たりが良くなる。
    自分側（マイクが別）は触らない。`self_name` の行は判定にかけない。
    """
    remote_rows = [row for row in rows if row.get("speaker") != self_name]
    if not remote_rows:
        return {}
    return label_segments(remote_rows, remote_audio, sample_rate, diarizer)


def gap_audio(gap: dict, audios: dict[str, "np.ndarray"], sample_rate: int):
    """取りこぼした区間の音声を録音から切り出す。"""
    audio = audios.get(str(gap.get("stream", "remote")))
    if audio is None:
        return None
    start, end = float(gap.get("start_time", 0.0)), float(gap.get("end_time", 0.0))
    piece = audio[int(start * sample_rate):int(end * sample_rate)]
    return piece if piece.size else None


def pick_up_gaps_external(session: Path, gaps: list[dict], cfg: ExternalSttConfig, sample_rate: int,
                          audios: dict, self_name: str, *, approved: bool) -> list[dict]:
    """外へ送れなかった区間だけを、もう一度外へ投げる。関門はここでも通す。"""
    files = [session / "recording_remote.wav", session / "recording_self.wav"]
    approval = approve(cfg, session, [path for path in files if path.exists()],
                       ask=(lambda prompt: True) if approved else None)
    record_send(session, approval, model=cfg.model, note="finalize_gaps")
    api = TranscribeApi(load_key(cfg.key_file),
                        TranscribeConfig(model=cfg.model, timeout_sec=cfg.timeout_sec))
    rows: list[dict] = []
    for gap in gaps:
        audio = gap_audio(gap, audios, sample_rate)
        if audio is None:
            continue
        start = float(gap.get("start_time", 0.0))
        result = api.transcribe_audio(audio, sample_rate, session / "external")
        for item in result.get("utterances", []):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            rows.append({"speaker": self_name if gap.get("stream") == "self" else "",
                         "text": text,
                         "start_time": round(start + float(item.get("start", 0.0) or 0.0), 2),
                         "end_time": round(start + float(item.get("end", 0.0) or 0.0), 2)})
    return rows


def pick_up_gaps_locally(gaps: list[dict], sample_rate: int, audios: dict, vad_cfg: VadConfig,
                         diarizer, whisper: WhisperClient, self_name: str, block_sec: float,
                         turn_cfg: TurnSplitConfig) -> list[dict]:
    """拾い直せなかった区間を、手元の Whisper で起こす。会議が失われてはいけない。"""
    rows: list[dict] = []
    for gap in gaps:
        audio = gap_audio(gap, audios, sample_rate)
        if audio is None:
            continue
        start = float(gap.get("start_time", 0.0))
        if gap.get("stream") == "self":
            found = transcribe_self(audio, sample_rate, vad_cfg, whisper, self_name, block_sec)
        else:
            found = [{"speaker": row["speaker"], "text": row["text"],
                      "start_time": row["start_time"], "end_time": row["end_time"]}
                     for row in replay(audio, sample_rate, vad_cfg, diarizer, whisper, block_sec, turn_cfg)
                     if row["text"]]
        for row in found:
            row["start_time"] = round(start + row["start_time"], 2)
            row["end_time"] = round(start + row["end_time"], 2)
        rows += found
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="セッションディレクトリ")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "settings.yaml")
    parser.add_argument("--client", default=None, help="タスク管理 のクライアント名（省略時は settings.yaml）")
    parser.add_argument("--external-stt", dest="external_stt", action="store_true", default=None,
                        help="外（Gemini）の文字起こしを使う。設定が on でも、実行のたびに画面で承認を取る")
    parser.add_argument("--no-external-stt", dest="external_stt", action="store_false",
                        help="外へは出さず、手元の Whisper だけで作り直す")
    parser.add_argument("--reuse-live", dest="reuse_live", action="store_true", default=None,
                        help="会議中に外で起こしたぶんを使い、同じ音声を送り直さない（方式④）。"
                             "話者だけ当て直し、送れなかった区間だけ拾い直す")
    parser.add_argument("--no-reuse-live", dest="reuse_live", action="store_false",
                        help="会議中のぶんは使わず、録音から全部作り直す")
    parser.add_argument("--keep-live-speakers", action="store_true",
                        help="会議中に付いた話者をそのまま使う（当て直さない）。"
                             "画面で名前を付けた会議は、会議中のラベルのほうが正しいことがある")
    parser.add_argument("--approved-in-ui", action="store_true",
                        help="画面で承認済み。人が画面で答えた直後に呼び出し側が付けるもので、"
                             "手で付けるものではない（課金の確認と送信の記録はそのまま通る）")
    parser.add_argument("--fresh-speakers", action="store_true",
                        help="speakers.json を引き継がず、声だけで一から分ける"
                             "（登録がうまくいかなかった回に使う。2026-09-11 は物音が 参加者B として登録されていた）")
    args = parser.parse_args()
    ffmpeg.require("録音からの作り直し")

    session = args.session
    remote_path = session / "recording_remote.wav"
    self_path = session / "recording_self.wav"
    if not remote_path.exists():
        raise SystemExit(f"録音が見つかりません: {remote_path}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    meeting = config.get("meeting", {})
    vad_values = {**config.get("vad", {}), **meeting.get("vad", {})}
    vad_cfg = VadConfig(**{k: v for k, v in vad_values.items() if k in VadConfig.__dataclass_fields__})
    turn_cfg = TurnSplitConfig(**{k: v for k, v in meeting.get("turn_split", {}).items() if k in TurnSplitConfig.__dataclass_fields__})
    diarizer_values = {**meeting.get("diarizer", {})}
    diarizer_cfg = DiarizerConfig(**{k: v for k, v in diarizer_values.items() if k in DiarizerConfig.__dataclass_fields__})
    self_name = meeting.get("self_name", "自分")
    block_sec = config.get("audio", {}).get("chunk_duration_sec", 0.1)

    stt_cfg = SttConfig(**config.get("stt", {}))
    if stt_cfg.min_avg_logprob is None:
        stt_cfg.min_avg_logprob = meeting.get("min_avg_logprob", -1.0)
    whisper = WhisperClient(stt_cfg)

    external_cfg = ExternalSttConfig.from_mapping(meeting.get("external_stt"))
    if args.external_stt is not None:
        external_cfg.enabled = args.external_stt

    enrollment_cfg = EnrollmentConfig(diarizer=diarizer_cfg)
    if args.fresh_speakers:
        diarizer, loaded = EnrolledDiarizer(diarizer_cfg), False
    else:
        diarizer, loaded = load_or_create(session / "speakers.json", enrollment_cfg)
    print(f"話者登録: {'引き継ぎ' if loaded else 'なし（声だけで分ける）'} {diarizer.enrolled_names}")

    sample_rate = int(config.get("audio", {}).get("sample_rate", 48000))

    # 外へ出す経路。関門を通らなければ（承認なし・オフライン・課金を確認できない）、
    #   何も送らずに手元の Whisper へ落ちる。会議が失われてはいけない。
    rows: list[dict] | None = None
    glossary_path = meeting.get("glossary_path", "config/glossary.yaml")

    # 方式④: 会議中に外で起こしていたら、同じ音声を送り直さない
    reuse_live = sent_live(session) if args.reuse_live is None else args.reuse_live
    live_rows = read_jsonl(session / "transcripts.jsonl") if reuse_live else []
    if reuse_live and not live_rows:
        print("  会議中の全文がありません。録音から作り直します。")
        reuse_live = False
    if reuse_live:
        glossary_path = external_cfg.glossary_path or glossary_path
        remote_audio = decode_audio(remote_path, sample_rate)
        audios = {"remote": remote_audio}
        if self_path.exists():
            audios["self"] = decode_audio(self_path, sample_rate)
        print(f"\n── 会議中に外で起こした {len(live_rows)} 行を使います（送り直しません）──")
        if args.keep_live_speakers:
            # 会議中に人が付けた名前は、当て直すと消えることがある（声紋に結び付いていない場合）。
            #   2026-09-14 の会議では、当て直しで 736 行の名前が「不明話者1・2」へ割れた。
            print("  話者は会議中のまま使います（当て直しません）")
        else:
            counts = relabel_live_rows(live_rows, remote_audio, sample_rate, diarizer, self_name)
            print(f"  話者を当て直しました: {counts}")

        gaps = read_jsonl(session / GAPS_FILE)
        if gaps:
            lost = sum(float(gap.get("end_time", 0)) - float(gap.get("start_time", 0)) for gap in gaps)
            print(f"  会議中に外へ送れなかった区間が {len(gaps)} 件（{lost:.0f} 秒）あります。拾い直します。")
            picked: list[dict] = []
            remaining = gaps
            if external_cfg.enabled:
                try:
                    picked = pick_up_gaps_external(session, gaps, external_cfg, sample_rate,
                                                   audios, self_name, approved=args.approved_in_ui)
                    remaining = []
                except ExternalSendRefused as refused:
                    print(f"  外へは出しません（{refused}）。手元で拾い直します。")
                except Exception as error:  # noqa: BLE001 — 何が起きても手元へ落ちる
                    print(f"  外での拾い直しに失敗しました（{error}）。手元で拾い直します。")
            if remaining:
                picked = pick_up_gaps_locally(remaining, sample_rate, audios, vad_cfg, diarizer,
                                              whisper, self_name, block_sec, turn_cfg)
            # 外で拾い直したぶんは話者が空のまま＝手元の声紋で当てる
            unlabeled = [row for row in picked if not row["speaker"]]
            if unlabeled:
                label_segments(unlabeled, remote_audio, sample_rate, diarizer)
            print(f"  拾い直しで {len(picked)} 行を足しました")
            live_rows += picked
        rows = sorted(({"speaker": row.get("speaker", ""), "text": row.get("text", ""),
                        "start_time": float(row.get("start_time", 0.0)),
                        "end_time": float(row.get("end_time", 0.0))}
                       for row in live_rows if row.get("text")),
                      key=lambda row: row["start_time"])

    if rows is None and external_cfg.enabled:
        try:
            rows = transcribe_external(session, remote_path, self_path, self_name,
                                       external_cfg, sample_rate, diarizer,
                                       approved=args.approved_in_ui)
            glossary_path = external_cfg.glossary_path or glossary_path
        except ExternalSendRefused as refused:
            print(f"\n  外へは出しません（{refused}）。手元の文字起こしで作り直します。")
        except Exception as error:  # noqa: BLE001 — 何が起きても手元へ落ちる
            print(f"\n  外の文字起こしに失敗しました（{error}）。手元の文字起こしで作り直します。")

    if rows is None:
        remote_audio = decode_audio(remote_path, sample_rate)
        print(f"\n── 相手側 {remote_audio.size / sample_rate / 60:.1f} 分 ──")
        remote_rows = [
            {"speaker": row["speaker"], "text": row["text"], "start_time": row["start_time"], "end_time": row["end_time"]}
            for row in replay(remote_audio, sample_rate, vad_cfg, diarizer, whisper, block_sec, turn_cfg)
            if row["text"]
        ]

        self_rows: list[dict] = []
        if self_path.exists():
            self_audio = decode_audio(self_path, sample_rate)
            print(f"\n── 自分側 {self_audio.size / sample_rate / 60:.1f} 分 ──")
            self_rows = transcribe_self(self_audio, sample_rate, vad_cfg, whisper, self_name, block_sec)

        rows = sorted(remote_rows + self_rows, key=lambda row: row["start_time"])

    # 画面で付けた名前を当て直す（声紋に結び付いていなかったぶんは、ここでしか戻らない）
    renames = read_jsonl(session / RENAMES_FILE)
    if renames:
        applied = apply_renames(rows, renames)
        print(f"  画面で付けた名前 {len(renames)} 件のうち {applied} 行に反映しました")

    glossary = load_glossary(REPO / glossary_path)
    if (config.get("task_hub") or {}).get("dictionary_sync"):
        # 会議中と同じ置き換えにする（タスク管理 の辞書の名前の対）
        glossary = task_hub_dictionary.merge_pairs(
            glossary, task_hub_dictionary.live_pairs(task_hub_dictionary.load_cache(REPO / task_hub_dictionary.CACHE_FILE)))
    protected = load_protected(REPO / glossary_path)
    # 会議中と同じ守り札にする: 事前資料の固有名詞と画面で付けた名前（作り直しで社名を壊し直さない）
    protected += load_terms(session)
    protected += [part for entry in renames for part in str(entry.get("new", "")).split()
                  if len(part) >= 3 and not part.startswith("不明話者")]
    for row in rows:
        row["text"] = apply_glossary(row["text"], glossary, protected)

    marked = mark_rows(rows, enabled=bool(meeting.get("aizuchi_mark", True)))
    if marked:
        print(f"相づちだけの行 {marked} 件に印を付けました（落としてはいません）")

    corrections_path = session / "corrections.jsonl"
    corrections = [json.loads(line) for line in corrections_path.read_text(encoding="utf-8").splitlines() if line.strip()] if corrections_path.exists() else []
    applied = apply_corrections(rows, [c for c in corrections if not c.get("auto")])
    print(f"\n会議中の訂正 {len(corrections)} 件のうち {applied} 件を反映しました")

    final_path = session / "transcripts_final.jsonl"
    with final_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "timestamp": ""}, ensure_ascii=False) + "\n")

    state_path = session / "state.json"
    state = MeetingState.from_json(state_path.read_text(encoding="utf-8")) if state_path.exists() else MeetingState()
    for name in ("minutes_input.md", "minutes_handoff.json"):
        source = session / name
        backup = source.with_suffix(f".live{source.suffix}")
        if source.exists() and not backup.exists():
            source.rename(backup)
    segments = [TranscriptSegment(row["speaker"], row["text"], row["start_time"], row["end_time"], "") for row in rows]
    session_info = {
        "client_name": args.client or config.get("task_hub", {}).get("client_name", "（未設定）"),
        "meeting_date": session.name[:10],
        "session_name": session.name,
    }
    _, handoff = write_minutes_handoff(session, state, segments, session_info, [], [],
                                       mark_aizuchi=bool(meeting.get("aizuchi_mark", True)))
    # 作り直しは古い speakers.json をそのまま引き継ぐので、失敗した登録が何度でも効く。
    #   会議の終わりと同じ検査をここでもする（2026-09-13 に繋ぎ忘れていて見逃した）。
    warnings = failed_enrollment_lines(diarizer, session / "speakers.json")
    if warnings:
        print()
        for line in warnings:
            print(line)

    print(f"\n  {len(rows)} 行を書き出しました: {final_path}")
    print(f"  議事録の材料を作り直しました: {handoff}")
    print("  要約も作り直すなら: python scripts/replay_state.py "
          f"{final_path} --session {session.name}-rebuilt --config {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
