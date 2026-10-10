"""Evaluate on REAL recordings: a folder of audio files with matching .txt
references (e.g. notes you dictate on your phone, or a FLEURS / Common Voice
export).

  data/recordings/
    note_001.wav   note_001.txt
    note_002.m4a   note_002.txt      # any format ffmpeg can read

  python scripts/eval_recordings.py data/recordings --models base-int8 small-int8

Prints WER / term recall / number metrics raw vs lexicon-corrected and writes
results/recordings_<model>.json. Uses the same normaliser, corrector and
threshold as the synthetic benchmark, so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_stt.asr import VadSegmenter, WhisperASR, transcribe_long  # noqa: E402
from care_stt.config import ROOT, SR  # noqa: E402
from care_stt.correct import LexiconCorrector  # noqa: E402
from care_stt.lexicon import Lexicon  # noqa: E402
from care_stt.metrics import aggregate, score_utt  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".aac"}


def read_audio(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--models", nargs="+", default=["base-int8"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="correction threshold (default: value tuned on dev in results/metrics.json)")
    args = ap.parse_args()

    th = args.threshold
    if th is None:
        mp = ROOT / "results/metrics.json"
        th = json.loads(mp.read_text())["threshold_tuning"]["chosen"] if mp.exists() else 0.75
    lex = Lexicon.load()
    corrector = LexiconCorrector(lex, th)
    pairs = [(a, a.with_suffix(".txt")) for a in sorted(Path(args.folder).iterdir())
             if a.suffix.lower() in AUDIO_EXT and a.with_suffix(".txt").exists()]
    if not pairs:
        raise SystemExit(f"no audio+txt pairs in {args.folder}")
    vad = VadSegmenter()
    for m in args.models:
        asr = WhisperASR(*m.split("-"))
        raw, cor, rows = [], [], []
        for a, t in pairs:
            ref = t.read_text(encoding="utf-8").strip()
            x = read_audio(a)
            tr = transcribe_long(asr, vad, x) if len(x) > 25 * SR else asr.transcribe(x)
            fixed, corrs = corrector.correct(tr.text)
            raw.append(score_utt(ref, tr.text, lex))
            cor.append(score_utt(ref, fixed, lex))
            rows.append({"file": a.name, "ref": ref, "hyp": tr.text, "corrected": fixed,
                         "changes": [[c.original, c.replacement] for c in corrs], "rtf": round(tr.rtf, 3)})
        res = {"model": m, "threshold": th, "n": len(pairs),
               "raw": aggregate(raw).as_dict(), "corrected": aggregate(cor).as_dict(), "rows": rows}
        out = ROOT / f"results/recordings_{m}.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        r, c = res["raw"], res["corrected"]
        print(f"{m}: n={len(pairs)}  WER {r['wer']:.3f}->{c['wer']:.3f}  terms {r['term_recall']:.3f}->"
              f"{c['term_recall']:.3f}  numbers(all right) {r['number_utt_acc']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
