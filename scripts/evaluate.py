"""Score all transcripts, tune the correction threshold on dev, report on test.

  python scripts/evaluate.py            # writes results/metrics.json, results/report.md, results/*.png

Threshold tuning: the lexicon corrector's threshold is chosen on the DEV split
only (different sentences, different voices) as the lowest threshold whose
corrections are >= 95% precise; that single value is then frozen and applied
to every test model and condition.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_stt.config import ROOT  # noqa: E402
from care_stt.correct import LexiconCorrector  # noqa: E402
from care_stt.data import load_corpus  # noqa: E402
from care_stt.degrade import CONDITIONS  # noqa: E402
from care_stt.extract import extract, score_extraction  # noqa: E402
from care_stt.lexicon import Lexicon  # noqa: E402
from care_stt.metrics import aggregate, score_utt  # noqa: E402
from care_stt.normalize import normalize  # noqa: E402

TR = ROOT / "results/transcripts"
OUT = ROOT / "results"
TARGET_PRECISION = 0.95


def load_rows(split: str, model: str, cond: str) -> list[dict] | None:
    p = TR / split / model / f"{cond}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge_corrections(corrector: LexiconCorrector, rows, corpus, lex) -> tuple[int, int]:
    """(good, bad) corrections. A correction is good if the term it inserts is
    in the reference more often than in the uncorrected hypothesis."""
    good = bad = 0
    for r in rows:
        ref_terms = Counter(lex.terms_in(normalize(corpus[r["id"]].text)))
        hyp_terms = Counter(lex.terms_in(normalize(r["hyp"])))
        _, corrs = corrector.correct(r["hyp"])
        for c in corrs:
            if ref_terms[c.replacement] > hyp_terms[c.replacement]:
                good += 1
                hyp_terms[c.replacement] += 1
            else:
                bad += 1
    return good, bad


def tune_threshold(corpus, lex) -> dict:
    rows = []
    for m in ("tiny-int8", "base-int8"):
        for cond in ("clean", "babble_10"):
            rows += load_rows("dev", m, cond) or []
    if not rows:
        raise SystemExit("no dev transcripts - run scripts/transcribe.py --split dev first")
    sweep = []
    for i in range(0, 15):
        th = round(0.60 + 0.025 * i, 3)
        g, b = judge_corrections(LexiconCorrector(lex, th), rows, corpus, lex)
        sweep.append({"threshold": th, "good": g, "bad": b, "precision": round(g / max(g + b, 1), 4)})
    ok = [s for s in sweep if s["precision"] >= TARGET_PRECISION and s["good"] > 0]
    chosen = min(ok, key=lambda s: s["threshold"]) if ok else max(sweep, key=lambda s: s["precision"])
    return {"chosen": chosen["threshold"], "target_precision": TARGET_PRECISION,
            "dev_rows": len(rows), "sweep": sweep}


def latency(rows) -> dict:
    d = sorted(r["decode_s"] for r in rows)
    rtf = [r["decode_s"] / r["audio_s"] for r in rows]
    return {"decode_p50_s": round(statistics.median(d), 3),
            "decode_p95_s": round(d[min(len(d) - 1, int(0.95 * len(d)))], 3),
            "rtf_mean": round(statistics.mean(rtf), 3),
            "audio_mean_s": round(statistics.mean(r["audio_s"] for r in rows), 2)}


def bootstrap(raw_scores, cor_scores, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% bootstrap CIs (resampling utterances) for WER and term recall, and
    for the correction's term-recall gain on the SAME resamples (paired)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(raw_scores)
    out = {"wer": [], "term_recall": [], "term_recall_corrected": [], "term_recall_gain": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = aggregate([raw_scores[i] for i in idx])
        b = aggregate([cor_scores[i] for i in idx])
        out["wer"].append(a.wer)
        out["term_recall"].append(a.term_recall)
        out["term_recall_corrected"].append(b.term_recall)
        out["term_recall_gain"].append(b.term_recall - a.term_recall)
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in out.items()}


def evaluate_rows(rows, corpus, lex, corrector, with_ci: bool = False) -> dict:
    raw_scores, cor_scores, ex_raw, ex_cor, golds = [], [], [], [], []
    good = bad = 0
    for r in rows:
        u = corpus[r["id"]]
        fixed, corrs = corrector.correct(r["hyp"])
        raw_scores.append(score_utt(u.text, r["hyp"], lex))
        cor_scores.append(score_utt(u.text, fixed, lex))
        ref_terms = Counter(lex.terms_in(normalize(u.text)))
        hyp_terms = Counter(lex.terms_in(normalize(r["hyp"])))
        for c in corrs:
            if ref_terms[c.replacement] > hyp_terms[c.replacement]:
                good += 1
                hyp_terms[c.replacement] += 1
            else:
                bad += 1
        golds.append({"vitals": u.vitals, "doses": u.doses})
        ex_raw.append(extract(r["hyp"], lex))
        ex_cor.append(extract(fixed, lex))
    res = {
        "raw": aggregate(raw_scores).as_dict(),
        "corrected": aggregate(cor_scores).as_dict(),
        "corrections": {"applied": good + bad, "good": good, "bad": bad,
                        "precision": round(good / max(good + bad, 1), 4)},
        "extraction_raw": score_extraction(golds, ex_raw),
        "extraction_corrected": score_extraction(golds, ex_cor),
        "latency": latency(rows),
    }
    if with_ci:
        res["ci95"] = bootstrap(raw_scores, cor_scores)
    return res


def main() -> None:
    lex = Lexicon.load()
    corpus = {u.id: u for u in load_corpus()}
    tuning = tune_threshold(corpus, lex)
    corrector = LexiconCorrector(lex, tuning["chosen"])
    results = {"threshold_tuning": tuning, "test": {}}
    models = sorted(p.name for p in (TR / "test").iterdir() if p.is_dir())
    for m in models:
        results["test"][m] = {}
        for cond in CONDITIONS:
            rows = load_rows("test", m, cond)
            if rows:
                results["test"][m][cond] = evaluate_rows(rows, corpus, lex, corrector, with_ci=(cond == "clean"))
    # file sizes for the deployment trade-off table
    from care_stt.config import MODELS_DIR
    sizes = {}
    for m in models:
        size, quant = m.split("-")
        d = MODELS_DIR / f"sherpa-onnx-whisper-{size}"
        sfx = ".int8.onnx" if quant == "int8" else ".onnx"
        files = [d / f"{size}-encoder{sfx}", d / f"{size}-decoder{sfx}"]
        if all(f.exists() for f in files):
            sizes[m] = round(sum(f.stat().st_size for f in files) / 1e6, 1)
    results["model_size_mb"] = sizes
    per_voice = {}
    for m in models:
        rows = load_rows("test", m, "clean") or []
        by = {}
        for r in rows:
            by.setdefault(r["voice"], []).append(score_utt(corpus[r["id"]].text, r["hyp"], lex))
        per_voice[m] = {v: aggregate(s).as_dict() for v, s in by.items()}
    results["per_voice_clean"] = per_voice
    lat = OUT / "latency.json"
    results["latency_idle"] = json.loads(lat.read_text()) if lat.exists() else {}
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_errors(corpus, lex, corrector)
    from report import write_report  # local module in scripts/
    write_report(results)
    print(f"threshold={tuning['chosen']}  -> results/metrics.json, results/report.md")


def write_errors(corpus, lex, corrector) -> None:
    """Worked examples for the README: what the corrector changed on test."""
    rows = load_rows("test", "base-int8", "clean") or []
    out = []
    for r in rows:
        fixed, corrs = corrector.correct(r["hyp"])
        if corrs:
            out.append({"id": r["id"], "voice": r["voice"], "ref": corpus[r["id"]].text,
                        "hyp": r["hyp"], "corrected": fixed,
                        "changes": [[c.original, c.replacement, c.score] for c in corrs]})
    (OUT / "corrections_base-int8_clean.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
