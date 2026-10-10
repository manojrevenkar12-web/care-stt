"""Structured extraction from a dictated care note, with review flags.

The transcript is only useful if the vitals and doses land in the right chart
fields. A wrong value that is written to the chart silently is the failure
that matters, so every extracted value is checked and anything doubtful is
flagged for the caregiver to confirm instead of being auto-filed:

  * implausible   - outside physiological range (pulse 22, temp 47,8)
  * inconsistent  - diastolic >= systolic
  * missing_value - the field name was heard but no value could be parsed
                    ("Blutdruck ... Puls 72": the reading itself was lost)
  * unknown_drug  - a dose ("... 5 Milligramm") with no recognised drug name
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import Lexicon
from .normalize import normalize

NUM = r"(\d+(?:,\d+)?)"
PATTERNS = {
    "bp": re.compile(rf"blutdruck {NUM} zu {NUM}"),
    "pulse": re.compile(rf"puls {NUM}"),
    "temp": re.compile(rf"temperatur {NUM}"),
    "spo2": re.compile(rf"sauerstoffsättigung {NUM}"),
    "glucose": re.compile(rf"blutzucker (?:[a-zäöü]+ ){{0,4}}?{NUM}"),
    "resp_rate": re.compile(rf"atemfrequenz {NUM}"),
}
TRIGGERS = {"bp": "blutdruck", "pulse": "puls", "temp": "temperatur", "spo2": "sauerstoffsättigung",
            "glucose": "blutzucker", "resp_rate": "atemfrequenz"}
RANGES = {"bp_sys": (60, 250), "bp_dia": (30, 150), "pulse": (30, 200), "temp": (34.0, 42.0),
          "spo2": (70, 100), "glucose": (30, 600), "resp_rate": (6, 50)}
DOSE = re.compile(rf"([a-zäöü]+) {NUM} milligramm")
INSULIN = re.compile(rf"{NUM} einheiten insulin")


def _num(s: str) -> float | int:
    return float(s.replace(",", ".")) if "," in s else int(s)


@dataclass
class Extraction:
    vitals: dict = field(default_factory=dict)
    doses: list = field(default_factory=list)       # [drug, amount, unit]
    flags: list = field(default_factory=list)        # [field, reason]

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)

    def flagged(self, name: str) -> bool:
        base = name.split("_")[0] if name.startswith("bp_") else name
        return any(f[0] in (name, base) for f in self.flags)


def extract(text: str, lexicon: Lexicon) -> Extraction:
    t = normalize(text)
    ex = Extraction()
    for name, pat in PATTERNS.items():
        m = pat.search(t)
        if not m:
            if re.search(rf"\b{TRIGGERS[name]}\b", t):
                ex.flags.append([name, "missing_value"])
            continue
        if name == "bp":
            ex.vitals["bp_sys"], ex.vitals["bp_dia"] = _num(m.group(1)), _num(m.group(2))
        else:
            ex.vitals[name] = _num(m.group(1))
    for k, v in ex.vitals.items():
        lo, hi = RANGES[k]
        if not lo <= v <= hi:
            ex.flags.append([k, "implausible"])
    if "bp_sys" in ex.vitals and ex.vitals["bp_dia"] >= ex.vitals["bp_sys"]:
        ex.flags.append(["bp", "inconsistent"])

    drugs = {lexicon.norm[c][0]: c for c in lexicon.canonical if len(lexicon.norm[c]) == 1}
    for m in DOSE.finditer(t):
        word, amount = m.group(1), _num(m.group(2))
        if word in drugs:
            ex.doses.append([drugs[word], amount, "mg"])
        else:
            ex.doses.append([None, amount, "mg"])
            ex.flags.append(["dose", "unknown_drug"])
    for m in INSULIN.finditer(t):
        ex.doses.append(["Insulin", _num(m.group(1)), "IE"])
    return ex


def score_extraction(golds: list[dict], preds: list[Extraction]) -> dict:
    """Field-level scoring over utterances that have gold structure.

    correct        - value extracted and equal to gold
    wrong_silent   - value extracted, wrong, and NOT flagged  (the dangerous case)
    wrong_flagged  - value extracted, wrong, but flagged for review
    missed         - no value extracted
    """
    c = {"fields": 0, "correct": 0, "wrong_silent": 0, "wrong_flagged": 0, "missed": 0,
         "correct_flagged": 0, "utts_needing_review": 0, "utts": 0}
    for g, p in zip(golds, preds):
        gold_fields = dict(g.get("vitals") or {})
        for d in g.get("doses") or []:
            gold_fields[f"dose:{d[0]}"] = d[1]
        if not gold_fields:
            continue
        c["utts"] += 1
        c["utts_needing_review"] += p.needs_review
        pred_fields = dict(p.vitals)
        for d in p.doses:
            if d[0]:
                pred_fields[f"dose:{d[0]}"] = d[1]
        unmatched_doses = [d for d in p.doses if not d[0]]
        for k, gv in gold_fields.items():
            c["fields"] += 1
            pv = pred_fields.get(k)
            if pv is None and k.startswith("dose:") and unmatched_doses:
                pv, flagged = unmatched_doses.pop(0)[1], True   # dose heard, drug lost -> flagged
                key_ok = False
            else:
                flagged = p.flagged(k if not k.startswith("dose:") else "dose")
                key_ok = True
            if pv is None:
                c["missed"] += 1
            elif key_ok and abs(float(pv) - float(gv)) < 1e-6:
                c["correct"] += 1
                c["correct_flagged"] += flagged
            elif flagged or p.needs_review:
                c["wrong_flagged"] += 1
            else:
                c["wrong_silent"] += 1
    f = max(c["fields"], 1)
    c.update({
        "field_accuracy": round(c["correct"] / f, 4),
        "silent_error_rate": round(c["wrong_silent"] / f, 4),
        "wrong_caught_share": round(c["wrong_flagged"] / max(c["wrong_flagged"] + c["wrong_silent"], 1), 4),
        "review_rate": round(c["utts_needing_review"] / max(c["utts"], 1), 4),
    })
    return c
