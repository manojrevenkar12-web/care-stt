"""CI smoke gate: score a small transcript slice and fail on gross regression.

Bounds are deliberately loose (whisper-tiny, 40 utterances): the job exists
to catch a broken pipeline - wrong sample rate, empty outputs, a normaliser
bug - not to measure quality. Quality gates run on the full benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_stt.data import load_corpus  # noqa: E402
from care_stt.lexicon import Lexicon  # noqa: E402
from care_stt.metrics import aggregate, score_utt  # noqa: E402

BOUNDS = {"clean": 0.75, "babble_10": 0.90, "telephone": 0.90}  # max WER for whisper-tiny (measured: 0.60 / 0.75 / 0.74)


def main(folder: str) -> int:
    lex, corpus = Lexicon.load(), {u.id: u for u in load_corpus()}
    failed = False
    for cond, max_wer in BOUNDS.items():
        rows = [json.loads(l) for l in (Path(folder) / f"{cond}.jsonl").read_text().splitlines()]
        empty = sum(not r["hyp"] for r in rows)
        a = aggregate([score_utt(corpus[r["id"]].text, r["hyp"], lex) for r in rows])
        ok = a.wer <= max_wer and empty <= len(rows) // 10
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {cond}: WER {a.wer:.3f} (max {max_wer}), empty {empty}/{len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
