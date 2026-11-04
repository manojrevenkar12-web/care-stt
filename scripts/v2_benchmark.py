"""v2 pipeline after training: export, tune the biasing strength on dev, benchmark on test, fit the
gate, write the report.

  python scripts/v2_benchmark.py export   --run ~/models/finetuned/base-care-XXXX
  python scripts/v2_benchmark.py tune     (biasing bonus on the dev facility, dev voices)
  python scripts/v2_benchmark.py dev      (dev renderings the gate is trained on)
  python scripts/v2_benchmark.py test     (held-out voices; v1 test + facility test)
  python scripts/v2_benchmark.py report   (metrics.json, gate.json, report.md)

Nothing tuned here looks at a test voice or a test facility.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_stt import eval2, gate, synth  # noqa: E402
from care_stt.config import MODELS_DIR  # noqa: E402

STOCK = str(Path(MODELS_DIR) / "sherpa-onnx-whisper-base")
FT = str(Path(MODELS_DIR) / "finetuned" / "base-care-export")
BONUS_FILE = eval2.OUT / "bias_bonus.json"


def bonus() -> float:
    return json.loads(BONUS_FILE.read_text())["bonus"] if BONUS_FILE.exists() else 2.0


def decoders(bias: bool = True) -> list[eval2.Decoder]:
    out = [eval2.Decoder("stock", STOCK, 1, False), eval2.Decoder("finetuned", FT, 1, False)]
    if bias:
        out += [eval2.Decoder("stock+bias", STOCK, 4, True), eval2.Decoder("finetuned+bias", FT, 4, True)]
    return out


def cmd_export(run: str) -> None:
    from care_stt import export, train

    m = train.load_finetuned(Path(run))
    out = export.export(m, "base", Path(FT))
    print({k: f"{v.stat().st_size / 1e6:.1f} MB" for k, v in out.items()})
    (Path(FT) / "SOURCE").write_text(str(run) + "\n")


def cmd_tune() -> None:
    """Pick the bonus that recovers most names and plan drugs on the dev facility without raising
    WER on its notes by more than two points over no biasing."""
    voices, conds = synth.DEV_VOICES[:2], ["clean", "babble_10"]
    results = {}
    for b in (0.0, 1.0, 2.0, 3.0, 4.0):
        name = f"tune-ft-bonus{b}"
        dec = eval2.Decoder(name, FT, 4 if b else 1, b > 0)
        path = eval2.OUT / "tune.jsonl"
        eval2.run([dec], ["facility-dev"], voices, conds, out=path, bonus=b)
        rows = [json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["decoder"] == name]
        meta = {it["id"]: it for it in eval2.items("facility-dev")}
        s = eval2.score(rows, eval2.scoring_lexicon(), meta)
        drug = sum(eval2.fold(meta[r["id"]]["drug"]) in eval2.fold(r["hyp"]) for r in rows) / len(rows)
        results[b] = {"wer": s["wer"], "name_recall": s["name_recall"], "drug_recall": round(drug, 4)}
        print(b, results[b])
    base_wer = results[0.0]["wer"]
    ok = {b: r for b, r in results.items() if b > 0 and r["wer"] <= base_wer + 0.02}
    best = max(ok or results, key=lambda b: results[b]["name_recall"] + results[b]["drug_recall"])
    eval2.OUT.mkdir(parents=True, exist_ok=True)
    BONUS_FILE.write_text(json.dumps({"bonus": best, "dev": {str(k): v for k, v in results.items()}}, indent=1) + "\n")
    print("bonus", best)


def cmd_dev() -> None:
    eval2.run([d for d in decoders() if d.name == "finetuned+bias"], ["v1-dev", "facility-dev"], synth.DEV_VOICES[:2],
              eval2.CONDITIONS, bonus=bonus())
    # the gate also sees greedy output of the same dev notes, so it can be applied to either decoder



def cmd_test() -> None:
    """Held-out voices. Every decoder on the facility notes; on the hand-written v1 notes only the
    greedy decoders (those notes name no resident, and beam search costs ~3x the CPU time)."""
    conds = ["clean", "babble_10"]
    eval2.run(decoders(), ["facility"], synth.TEST_VOICES, conds, bonus=bonus())
    eval2.run(decoders(bias=False), ["v1"], synth.TEST_VOICES, conds, bonus=bonus())


def cmd_report() -> None:
    res = eval2.report()
    g = gate.evaluate("finetuned+bias")
    print(json.dumps({k: res[k]["facility"]["all"] for k in res if "facility" in res[k]}, indent=1))
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export", "tune", "dev", "test", "report"])
    ap.add_argument("--run")
    a = ap.parse_args()
    {"export": lambda: cmd_export(a.run), "tune": cmd_tune, "dev": cmd_dev, "test": cmd_test, "report": cmd_report}[a.stage]()
