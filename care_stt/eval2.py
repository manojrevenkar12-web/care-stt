"""v2 benchmark: stock vs fine-tuned decoder, with and without facility biasing, against the v1
post-correction baselines, on the deployed int8 graphs.

Test sets (held-out voices only): the v1 test split (hand-written notes and dictations) and the
facility test set (residents and medication plans the model never saw). Each utterance carries the
transcript, timings and per-token log-probabilities to results/v2/utterances.jsonl, which the
quality gate and the review console read.
"""
from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import corpus2, data, degrade, synth
from .config import CACHE_DIR, MODELS_DIR, ROOT
from .correct import LexiconCorrector
from .extract import extract, score_extraction
from .lexicon import Lexicon
from .metrics import aggregate, score_utt
from .runtime import Runtime

OUT = ROOT / "results" / "v2"
CONDITIONS = ["clean", "babble_10", "telephone"]


@dataclass(frozen=True)
class Decoder:
    name: str
    model_dir: str  # where the decoder graph lives
    beam: int
    bias: bool


def fold(s: str) -> str:
    """Case and diacritics folded: 'Erdoğan' and 'Erdogan' name the same resident."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def items(test_set: str) -> list[dict]:
    if test_set == "v1":
        return [dict(u.__dict__) for u in data.load_corpus(split="test")]
    if test_set == "v1-dev":
        return [dict(u.__dict__) for u in data.load_corpus(split="dev")]
    return corpus2.load({"facility": "facility_test.jsonl", "facility-dev": "facility_dev.jsonl"}[test_set])


def bias_for(item: dict) -> list[str]:
    if item.get("facility"):
        return corpus2.bias_list(item["facility"])
    return Lexicon.load().canonical[:44]  # notes without a facility: the drug catalogue the app ships with


def audio(item: dict, voice: str, condition: str) -> np.ndarray:
    p = CACHE_DIR / "v2audio" / condition / voice / f"{item['id']}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    rate = synth.speaking_rate(item["id"], voice, "test" if item["split"] == "test" else "dev")
    clean_p = CACHE_DIR / "v2audio" / "clean" / voice / f"{item['id']}.npy"
    if clean_p.exists():
        x = np.load(clean_p).astype(np.float32)
    else:
        x = synth.synthesize(item["text"], voice, rate)
        clean_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(clean_p, x.astype(np.float16))
    if condition == "clean":
        return x
    pool = data.load_corpus(split="dev")
    srcs = data.babble_sources(data.Utterance(id=item["id"], text="", split="test", kind=""), pool) if condition.startswith("babble") else []
    y = degrade.apply(condition, x, srcs, synth.seed_of(item["id"], voice, condition))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, y.astype(np.float16))
    return y


def run(decoders: list[Decoder], test_sets: list[str], voices: list[str], conditions: list[str],
        out: Path = OUT / "utterances.jsonl", bonus: float = 2.0, log=print) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l)
            done.add((r["decoder"], r["set"], r["id"], r["voice"], r["condition"]))
    runtimes = {}
    stock_enc = Path(MODELS_DIR) / "sherpa-onnx-whisper-base"
    with out.open("a", encoding="utf-8") as fh:
        for dec in decoders:
            rt = runtimes.setdefault(dec.model_dir, Runtime(dec.model_dir, "base", "int8", threads=2, encoder_dir=stock_enc))
            t0, n = time.time(), 0
            for ts in test_sets:
                for it in items(ts):
                    for v in voices:
                        for c in conditions:
                            key = (dec.name, ts, it["id"], v, c)
                            if key in done:
                                continue
                            r = rt.transcribe(audio(it, v, c), bias=bias_for(it) if dec.bias else None, beam=dec.beam, bonus=bonus)
                            fh.write(json.dumps({"decoder": dec.name, "set": ts, "id": it["id"], "voice": v, "condition": c,
                                                 "ref": it["text"], "hyp": r.text, "biased": r.biased,
                                                 "logprobs": [round(x, 3) for x in r.logprobs], "seconds": round(r.seconds, 3),
                                                 "audio_seconds": round(r.audio_seconds, 2)}, ensure_ascii=False) + "\n")
                            fh.flush()
                            n += 1
                            if n % 50 == 0:
                                log(f"{dec.name}: {n} utterances, {(time.time() - t0) / n:.2f} s each")
    return out


# ---------------------------------------------------------------- scoring

def scoring_lexicon() -> Lexicon:
    """Every drug and nursing term any test set can contain, so term recall covers facility drugs too."""
    plans = [d for f in corpus2.FACILITIES.values() for d in f["plan"]]
    return Lexicon(Lexicon.load().canonical + plans + corpus2.NEW_DRUGS)


def correctors() -> dict[str, callable]:
    glob = LexiconCorrector(Lexicon.load())
    per_fac = {f: LexiconCorrector(Lexicon(Lexicon.load().canonical + corpus2.bias_list(f)))
               for f in {**corpus2.FACILITIES, **corpus2.DEV_FACILITY}}
    return {"lexicon": lambda text, item: glob.correct(text)[0],
            "facility": lambda text, item: (per_fac[item["facility"]] if item.get("facility") else glob).correct(text)[0]}


def score(rows: list[dict], lex: Lexicon, meta: dict[str, dict]) -> dict:
    sc = [score_utt(r["ref"], r["hyp"], lex) for r in rows]
    ag = aggregate(sc).as_dict()
    out = {k: ag[k] for k in ("n", "wer", "term_recall", "n_terms", "number_recall", "number_utt_acc")}
    named = [r for r in rows if meta[r["id"]].get("resident")]
    if named:
        out["name_recall"] = round(sum(fold(meta[r["id"]]["resident"]) in fold(r["hyp"]) for r in named) / len(named), 4)
        for cls in ("train", "heldout", "new"):
            sub = [r for r in named if meta[r["id"]]["drug_class"] == cls]
            if sub:
                out[f"drug_recall_{cls}"] = round(sum(fold(meta[r["id"]]["drug"]) in fold(r["hyp"]) for r in sub) / len(sub), 4)
    golds = [meta[r["id"]] for r in rows if meta[r["id"]].get("vitals") or meta[r["id"]].get("doses")]
    preds = [extract(r["hyp"], lex) for r in rows if meta[r["id"]].get("vitals") or meta[r["id"]].get("doses")]
    if golds:
        e = score_extraction(golds, preds)
        out["fields_correct"] = round(e["correct"] / max(1, e["fields"]), 4)
        out["fields_wrong_silent"] = round(e["wrong_silent"] / max(1, e["fields"]), 4)
    rtf = [r["seconds"] / r["audio_seconds"] for r in rows]
    out["rtf_mean"] = round(float(np.mean(rtf)), 3)
    out["rtf_p95"] = round(float(np.percentile(rtf, 95)), 3)
    return out


def report(path: Path = OUT / "utterances.jsonl") -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    meta = {it["id"]: it for ts in ("v1", "facility", "v1-dev", "facility-dev") for it in items(ts)}
    lex, cor = scoring_lexicon(), correctors()
    # post-correction systems are computed from the greedy transcripts, not decoded again
    systems: dict[str, list[dict]] = {}
    for r in rows:
        systems.setdefault(r["decoder"], []).append(r)
        if r["decoder"] in ("stock", "finetuned"):
            for cname, f in cor.items():
                systems.setdefault(f"{r['decoder']}+{cname}-corrector", []).append({**r, "hyp": f(r["hyp"], meta[r["id"]])})
    res: dict = {}
    for sname, srows in systems.items():
        for ts in sorted({r["set"] for r in srows}):
            for c in ["all"] + sorted({r["condition"] for r in srows}):
                sel = [r for r in srows if r["set"] == ts and (c == "all" or r["condition"] == c)]
                if sel:
                    res.setdefault(sname, {}).setdefault(ts, {})[c] = score(sel, lex, meta)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    return res
