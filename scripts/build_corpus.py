"""Build the care-documentation evaluation corpus (data/corpus.jsonl).

Two sources:
  * data/narrative.txt  - hand-written free-text care notes (terms, meds, events)
  * templated dictations - vital signs and medication doses with randomised but
    clinically plausible values. Each carries gold structured fields so the
    downstream extraction can be scored, not just the transcript.

Deterministic (fixed seed). Split: 1/3 dev (used only to tune the correction
threshold), 2/3 test (reported numbers). Narrative and templated items are
split separately so both splits contain both kinds.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = random.Random(20261010)

MEDS = ["Ramipril", "Metformin", "Bisoprolol", "Torasemid", "Amlodipin", "Melperon",
        "Pipamperon", "Quetiapin", "Mirtazapin", "Tramadol", "Tilidin", "Oxycodon",
        "Prednisolon", "Allopurinol", "Gabapentin", "Pregabalin", "Sertralin",
        "Citalopram", "Memantin", "Furosemid", "Ibuprofen", "Metamizol", "Enoxaparin",
        "Ciprofloxacin", "Candesartan"]
DOSES = [2.5, 5, 10, 20, 25, 40, 50, 75, 100, 200, 400, 500]
TIMES = ["sieben", "acht", "zwölf", "achtzehn", "zwanzig", "zweiundzwanzig"]
TIME_NUM = {"sieben": 7, "acht": 8, "zwölf": 12, "achtzehn": 18, "zwanzig": 20, "zweiundzwanzig": 22}


def fmt(x: float) -> str:
    """German decimal formatting: 37.8 -> '37,8', 5.0 -> '5'."""
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", ",")


def templated() -> list[dict]:
    items = []
    for _ in range(12):
        s, d, p = rng.randint(95, 185), rng.randint(55, 105), rng.randint(48, 118)
        items.append({"text": f"Blutdruck {s} zu {d}, Puls {p}.",
                      "vitals": {"bp_sys": s, "bp_dia": d, "pulse": p}})
    for _ in range(8):
        t = round(rng.uniform(35.8, 39.6), 1)
        o = rng.randint(84, 99)
        items.append({"text": f"Temperatur {fmt(t)} Grad, Sauerstoffsättigung {o} Prozent.",
                      "vitals": {"temp": t, "spo2": o}})
    for _ in range(8):
        bz, ie = rng.randint(62, 340), rng.randint(2, 18)
        items.append({"text": f"Blutzucker vor dem Mittagessen {bz}, {ie} Einheiten Insulin gespritzt.",
                      "vitals": {"glucose": bz}, "doses": [["Insulin", ie, "IE"]]})
    for _ in range(6):
        af, p, s, d = rng.randint(12, 30), rng.randint(55, 110), rng.randint(100, 170), rng.randint(60, 100)
        items.append({"text": f"Atemfrequenz {af} pro Minute, Puls {p}, Blutdruck {s} zu {d}.",
                      "vitals": {"resp_rate": af, "pulse": p, "bp_sys": s, "bp_dia": d}})
    for _ in range(16):
        med, dose, t = rng.choice(MEDS), rng.choice(DOSES), rng.choice(TIMES)
        items.append({"text": f"{med} {fmt(dose)} Milligramm um {t} Uhr gegeben.",
                      "doses": [[med, dose, "mg"]], "time_h": TIME_NUM[t]})
    return items


def main() -> None:
    narrative = [{"text": l.strip()} for l in (ROOT / "data/narrative.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    tmpl = templated()
    out = []
    for group, prefix in ((narrative, "n"), (tmpl, "v")):
        idx = list(range(len(group)))
        rng.shuffle(idx)
        n_dev = len(group) // 3
        for rank, i in enumerate(idx):
            item = dict(group[i])
            item["id"] = f"{prefix}{i:03d}"
            item["split"] = "dev" if rank < n_dev else "test"
            item["kind"] = "narrative" if prefix == "n" else "dictation"
            out.append(item)
    out.sort(key=lambda r: r["id"])
    path = ROOT / "data/corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_test = sum(r["split"] == "test" for r in out)
    print(f"wrote {len(out)} utterances ({len(out) - n_test} dev / {n_test} test) -> {path}")


if __name__ == "__main__":
    main()
