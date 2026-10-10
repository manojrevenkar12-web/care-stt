"""Latency benchmark on an otherwise idle machine.

Decode times inside the accuracy grid can be disturbed by other work, so
latency is measured separately: each model decodes the same 40 clean test
utterances after a warm-up, at 1 thread (phone-like budget) and 2 threads.

  python scripts/latency.py --models tiny-int8 tiny-fp32 base-int8 base-fp32 small-int8 small-fp32
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_stt.asr import WhisperASR  # noqa: E402
from care_stt.config import ROOT  # noqa: E402
from care_stt.data import audio, load_corpus  # noqa: E402
from care_stt.synth import TEST_VOICES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny-int8", "base-int8", "small-int8"])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    utts = load_corpus(split="test")[: args.n // 2]
    clips = [audio(u, v, "clean", []) for u in utts for v in TEST_VOICES]
    out_path = ROOT / "results/latency.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for m in args.models:
        for th in args.threads:
            asr = WhisperASR(*m.split("-"), threads=th)
            asr.transcribe(clips[0])  # warm-up
            ts = [asr.transcribe(c) for c in clips]
            d = sorted(t.seconds for t in ts)
            res[f"{m}@{th}t"] = {
                "model": m, "threads": th, "n": len(ts), "size_mb": asr.size_mb,
                "audio_mean_s": round(statistics.mean(t.audio_seconds for t in ts), 2),
                "decode_p50_s": round(statistics.median(d), 3),
                "decode_p95_s": round(d[int(0.95 * (len(d) - 1))], 3),
                "rtf_mean": round(statistics.mean(t.rtf for t in ts), 3),
            }
            print(m, th, res[f"{m}@{th}t"], flush=True)
            out_path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
