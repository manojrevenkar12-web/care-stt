"""Transcribe the corpus under each acoustic condition with each model.

  python scripts/transcribe.py --split test --models tiny-int8 base-int8 small-int8 --conditions all
  python scripts/transcribe.py --split test --models base-int8 --conditions clean --limit 20   # CI smoke

Writes results/transcripts/<split>/<model>/<condition>.jsonl (one row per
utterance x voice). Existing complete files are skipped, so runs resume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_stt import synth  # noqa: E402
from care_stt.asr import WhisperASR  # noqa: E402
from care_stt.config import ROOT  # noqa: E402
from care_stt.data import audio, load_corpus  # noqa: E402
from care_stt.degrade import CONDITIONS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--models", nargs="+", default=["base-int8"])
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "results/transcripts"))
    args = ap.parse_args()

    conds = CONDITIONS if args.conditions == ["all"] else args.conditions
    utts = load_corpus(split=args.split)
    if args.limit:
        utts = utts[: args.limit]
    pool = load_corpus(split="dev")
    voices = synth.voices_for(args.split)
    expected = len(utts) * len(voices)

    for m in args.models:
        size, quant = m.split("-")
        asr = None
        for cond in conds:
            path = Path(args.out) / args.split / m / f"{cond}.jsonl"
            if path.exists() and sum(1 for _ in path.open()) == expected:
                print(f"skip {path} (complete)")
                continue
            asr = asr or WhisperASR(size, quant)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for u in utts:
                    for v in voices:
                        tr = asr.transcribe(audio(u, v, cond, pool))
                        f.write(json.dumps({"id": u.id, "voice": v, "hyp": tr.text,
                                            "decode_s": round(tr.seconds, 4),
                                            "audio_s": round(tr.audio_seconds, 3)}, ensure_ascii=False) + "\n")
            print(f"{m:12s} {cond:10s} -> {path}", flush=True)


if __name__ == "__main__":
    main()
