"""The v2 corpora: text to fine-tune on, and a facility test set the fine-tuned model has never seen.

What a model must get right in a care home is mostly words it cannot have learned in advance: the
names of the residents and the drugs on this facility's medication plan. So the data are split on
exactly those axes, and the split is checked by tests (tests/test_corpus2.py):

  train text      generated from phrase pools written for training only: training drugs, training
                  resident names, all nursing terms, its own sentence templates
  dev             the v1 dev split (hand-written notes and dictations), used for early stopping
  v1 test         the v1 test split (hand-written notes and dictations), unchanged, for continuity
  facility test   three fictional care homes, each with 12 residents and a medication plan. Test
                  templates are phrased differently from training templates; the residents' names
                  never appear in training; the plans use held-out lexicon drugs and drugs that are
                  in no list the model or the corrector was ever given

Voices are split as in v1: four for training and dev, two held out for every test set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .config import ROOT

LEXICON_DRUGS = [
    "Ramipril", "Metformin", "Pantoprazol", "Bisoprolol", "Torasemid", "Furosemid", "Simvastatin",
    "Novaminsulfon", "Metamizol", "Paracetamol", "Ibuprofen", "Amlodipin", "Candesartan", "Melperon",
    "Pipamperon", "Quetiapin", "Mirtazapin", "Lorazepam", "Tramadol", "Tilidin", "Oxycodon", "Fentanyl",
    "Prednisolon", "Clopidogrel", "Apixaban", "Rivaroxaban", "Allopurinol", "Levodopa", "Donepezil",
    "Memantin", "Citalopram", "Sertralin", "Pregabalin", "Gabapentin", "Levothyroxin", "Salbutamol",
    "Macrogol", "Lactulose", "Insulin", "Heparin", "Enoxaparin", "Amoxicillin", "Ciprofloxacin", "Nitrospray",
]
# never in training text: only a facility's medication plan (contextual biasing) can bring them in
HELDOUT_DRUGS = ["Metamizol", "Pipamperon", "Tilidin", "Oxycodon", "Rivaroxaban", "Allopurinol", "Memantin",
                 "Citalopram", "Sertralin", "Gabapentin", "Ciprofloxacin", "Clopidogrel"]
TRAIN_DRUGS = [d for d in LEXICON_DRUGS if d not in HELDOUT_DRUGS and d not in ("Insulin", "Nitrospray")]
# in no list at all: not in the lexicon the v1 corrector uses, not in training text
NEW_DRUGS = ["Edoxaban", "Dapagliflozin", "Empagliflozin", "Sitagliptin", "Tamsulosin", "Hydromorphon",
             "Risperidon", "Haloperidol", "Venlafaxin", "Duloxetin", "Spironolacton", "Nebivolol",
             "Atorvastatin", "Esomeprazol", "Ondansetron", "Valsartan"]

NURSING = ["Dekubitusprophylaxe", "Lagerung", "Mobilisation", "Inkontinenzmaterial", "Trinkprotokoll",
           "Bilanzierung", "Sturzprotokoll", "Wundversorgung", "Verbandwechsel", "Hautrötung",
           "Kontrakturenprophylaxe", "Pneumonieprophylaxe", "Thromboseprophylaxe", "Kompressionsstrümpfe",
           "Blasenkatheter", "Ernährungssonde", "Schaumverband", "Hydrogel", "Braden-Skala", "Exsikkose",
           "Ödeme", "Dyspnoe", "Obstipation", "Schluckstörung", "Rollator", "Bedarfsmedikation",
           "Physiotherapie", "Desorientierung"]

TRAIN_NAMES = ["Becker", "Schneider", "Wagner", "Fischer", "Meyer", "Krause", "Lehmann", "Schulz",
               "Kowalski", "Nowak", "Demir", "Kaya", "Petrov", "Ivanova", "Huber", "Gruber", "Vogel",
               "Keller", "Jansen", "Kühn", "Marković", "Horvat", "Öztürk", "Wolff"]
# residents of the three test facilities: none of these names occurs in training
FACILITIES = {
    "Haus Lindenhof": {
        "residents": ["Brandstätter", "Wiśniewski", "Yılmaz", "Hohenadel", "Zielinski", "Ostermann",
                      "Bäumler", "Kasprzak", "Scheuermann", "Dragomir", "Pfeiffer", "Szymańska"],
        "plan": ["Edoxaban", "Metamizol", "Tamsulosin", "Memantin", "Ramipril", "Pantoprazol",
                 "Haloperidol", "Gabapentin", "Levothyroxin", "Dapagliflozin"],
    },
    "Seniorenresidenz Am Park": {
        "residents": ["Achterberg", "Nguyen", "Lindqvist", "Ciobanu", "Rüdiger", "Hartwig", "Mazur",
                      "Eichhorn", "Tschernow", "Brockmann", "Aksoy", "Vollmer"],
        "plan": ["Empagliflozin", "Tilidin", "Risperidon", "Rivaroxaban", "Bisoprolol", "Metformin",
                 "Venlafaxin", "Allopurinol", "Torasemid", "Esomeprazol"],
    },
    "Pflegezentrum Nordufer": {
        "residents": ["Grabowski", "Feldkamp", "Sorokina", "Weißenborn", "Celik", "Ambrosius", "Kranich",
                      "Lewandowska", "Hamdi", "Stolzenberg", "Petrescu", "Wendland"],
        "plan": ["Hydromorphon", "Citalopram", "Spironolacton", "Clopidogrel", "Amlodipin", "Oxycodon",
                 "Nebivolol", "Sertralin", "Ciprofloxacin", "Ondansetron"],
    },
}

# tunes the biasing strength; shares no resident or drug with the test facilities or training
DEV_FACILITY = {
    "Haus Sonnenhang": {
        "residents": ["Kaltenbach", "Wróbel", "Erdoğan", "Lindemann", "Popescu", "Quandt", "Sarközi",
                      "Thalheim", "Jablonski", "Moritzen", "Okafor", "Reinwald"],
        "plan": ["Lercanidipin", "Mirabegron", "Bupropion", "Trimipramin", "Zopiclon", "Dabigatran",
                 "Pramipexol", "Moxonidin", "Ramipril", "Metformin"],
    },
}

DOSES = [2.5, 5, 10, 20, 25, 40, 50, 75, 100, 200, 400, 500]
TIMES = {"sieben": 7, "acht": 8, "zwölf": 12, "achtzehn": 18, "zwanzig": 20, "zweiundzwanzig": 22}


def fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", ",")


def _person(rng: random.Random, names: list[str]) -> tuple[str, str]:
    """('Frau Becker', 'Becker') with the German declension a nurse would use."""
    n = rng.choice(names)
    return (rng.choice(["Frau", "Herr"]) + " " + n, n)


def _dative(p: str) -> str:
    return p.replace("Herr ", "Herrn ")


# ------------------------------------------------------------------ training text

def _train_dictation(rng: random.Random) -> dict:
    k = rng.randrange(7)
    if k == 0:
        s, d, p = rng.randint(95, 185), rng.randint(55, 105), rng.randint(48, 118)
        t = rng.choice([f"Blutdruck {s} zu {d}, Puls {p}.", f"RR {s} zu {d}, Herzfrequenz {p}.",
                        f"Vitalzeichen: Blutdruck {s} zu {d}, Puls {p} pro Minute."])
        return {"text": t, "vitals": {"bp_sys": s, "bp_dia": d, "pulse": p}}
    if k == 1:
        t_, o = round(rng.uniform(35.8, 39.6), 1), rng.randint(84, 99)
        return {"text": rng.choice([f"Temperatur {fmt(t_)} Grad, Sauerstoffsättigung {o} Prozent.",
                                    f"Körpertemperatur {fmt(t_)}, Sättigung {o} Prozent."]),
                "vitals": {"temp": t_, "spo2": o}}
    if k == 2:
        bz, ie = rng.randint(62, 340), rng.randint(2, 18)
        meal = rng.choice(["dem Frühstück", "dem Mittagessen", "dem Abendessen"])
        return {"text": f"Blutzucker vor {meal} {bz}, {ie} Einheiten Insulin gespritzt.",
                "vitals": {"glucose": bz}, "doses": [["Insulin", ie, "IE"]]}
    if k == 3:
        med, dose, tw = rng.choice(TRAIN_DRUGS), rng.choice(DOSES), rng.choice(list(TIMES))
        return {"text": rng.choice([f"{med} {fmt(dose)} Milligramm um {tw} Uhr gegeben.",
                                    f"Um {tw} Uhr {med} {fmt(dose)} Milligramm verabreicht."]),
                "doses": [[med, dose, "mg"]], "time_h": TIMES[tw]}
    if k == 4:
        p, _ = _person(rng, TRAIN_NAMES)
        med, dose = rng.choice(TRAIN_DRUGS), rng.choice(DOSES)
        return {"text": rng.choice([f"{p} hat {med} {fmt(dose)} Milligramm bekommen.",
                                    f"{med} {fmt(dose)} Milligramm für {p} gerichtet und gegeben."]),
                "doses": [[med, dose, "mg"]]}
    if k == 5:
        p, _ = _person(rng, TRAIN_NAMES)
        s, d = rng.randint(95, 185), rng.randint(55, 105)
        return {"text": f"Bei {_dative(p)} Blutdruck {s} zu {d} gemessen.", "vitals": {"bp_sys": s, "bp_dia": d}}
    af, p = rng.randint(12, 30), rng.randint(55, 110)
    return {"text": f"Atemfrequenz {af} pro Minute, Puls {p}.", "vitals": {"resp_rate": af, "pulse": p}}


_SUBJ = ["Bewohnerin", "Bewohner", "Die Bewohnerin", "Der Bewohner"]
_OPEN = ["Frühdienst", "Spätdienst", "Nachtdienst", "Nachmittag", "Grundpflege", "Nach dem Mittagessen"]
_EVENTS = [
    "{term} nach Plan durchgeführt", "{term} dokumentiert, keine Auffälligkeiten",
    "{s} bei {term} unterstützt", "Kontrolle {term} ohne Befund", "{term} mit der Pflegefachkraft besprochen",
    "{s} lehnt {term} heute ab", "{term} wird fortgeführt", "Thema {term} an die Frühschicht übergeben",
]
_DRUG_EVENTS = [
    "{drug} wie verordnet gegeben", "{drug} nach Rücksprache mit dem Hausarzt pausiert",
    "{s} hat {drug} ohne Probleme eingenommen", "{drug} als Bedarfsmedikation angeboten",
    "Hausarzt hat {drug} neu angesetzt", "{drug} wird ab morgen reduziert",
]
_TAILS = ["", ", keine Besonderheiten", ", Angehörige informiert", ", weiter beobachten",
          ", Hausarzt wird informiert", ", Wohlbefinden gut"]


def _train_narrative(rng: random.Random) -> dict:
    s = rng.choice(_SUBJ)
    if rng.random() < 0.3:
        s, _ = _person(rng, TRAIN_NAMES)
    if rng.random() < 0.55:
        ev = rng.choice(_DRUG_EVENTS).format(drug=rng.choice(TRAIN_DRUGS), s=s)
    else:
        ev = rng.choice(_EVENTS).format(term=rng.choice(NURSING), s=s)
    # a shift or time opener the way notes are often headed ("Spätdienst: ...")
    text = (rng.choice(_OPEN) + ": " + ev) if rng.random() < 0.4 else ev
    return {"text": text[0].upper() + text[1:] + rng.choice(_TAILS) + "."}


def train_corpus(n: int = 1600, seed: int = 20261104) -> list[dict]:
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        item = _train_dictation(rng) if rng.random() < 0.5 else _train_narrative(rng)
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        item.update(id=f"t{len(out):04d}", split="train", kind="dictation" if "vitals" in item or "doses" in item else "narrative")
        out.append(item)
    return out


# ------------------------------------------------------------------ facility test set

_TEST_TEMPLATES = [
    ("{p} erhielt um {tw} Uhr {drug} {dose} Milligramm.", "dose_time"),
    ("Für {p} {drug} {dose} Milligramm laut Medikationsplan gestellt.", "dose"),
    ("{drug} {dose} Milligramm bei {pd} verabreicht, gut vertragen.", "dose"),
    ("{p}: Blutdruck {sys} zu {dia}, danach {drug} gegeben.", "bp_drug"),
    ("{p} klagt über Schwindel, {drug} vorerst abgesetzt.", "drug"),
    ("Angehörige von {pd} fragen nach {drug}, Rückruf zugesagt.", "drug"),
]


def facility_test(per_facility: int = 40, seed: int = 4711, facilities: dict | None = None, split: str = "test",
                  prefix: str = "f") -> list[dict]:
    rng = random.Random(seed)
    out = []
    for fname, f in (facilities or FACILITIES).items():
        for _ in range(per_facility):
            tpl, kind = rng.choice(_TEST_TEMPLATES)
            name = rng.choice(f["residents"])
            p = rng.choice(["Frau", "Herr"]) + " " + name
            drug, dose, tw = rng.choice(f["plan"]), rng.choice(DOSES), rng.choice(list(TIMES))
            sys_, dia = rng.randint(95, 185), rng.randint(55, 105)
            text = tpl.format(p=p, pd=_dative(p), drug=drug, dose=fmt(dose), tw=tw, sys=sys_, dia=dia)
            item = {"id": f"{prefix}{len(out):03d}", "split": split, "kind": "facility", "facility": fname,
                    "text": text, "resident": name, "drug": drug,
                    "drug_class": "new" if drug in NEW_DRUGS else "heldout" if drug in HELDOUT_DRUGS else "train"}
            if kind in ("dose", "dose_time"):
                item["doses"] = [[drug, dose, "mg"]]
            if kind == "dose_time":
                item["time_h"] = TIMES[tw]
            if kind == "bp_drug":
                item["vitals"] = {"bp_sys": sys_, "bp_dia": dia}
            out.append(item)
    return out


def bias_list(facility: str) -> list[str]:
    """What a facility's app would know before anyone speaks: its residents and its medication plan."""
    f = {**FACILITIES, **DEV_FACILITY}[facility]
    return f["residents"] + f["plan"]


def write(root: Path = ROOT / "data") -> None:
    dev_fac = facility_test(per_facility=40, seed=815, facilities=DEV_FACILITY, split="dev", prefix="g")
    for name, rows in (("train.jsonl", train_corpus()), ("facility_test.jsonl", facility_test()),
                       ("facility_dev.jsonl", dev_fac)):
        with (root / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (root / "facilities.json").write_text(json.dumps({**FACILITIES, **DEV_FACILITY}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    write()
