"""The repeat-or-confirm gate: decide, from the decoder's own signals, whether a note should go to
the chart as heard or be read back to the nurse first.

A misheard dose is worse than a slow note. The app can afford to ask "please check: Edoxaban 20
milligrams?" on a fraction of notes, so the question is which fraction catches most of the notes
that contain a chart error. The gate is a logistic regression (PyTorch) over features the runtime
already has: per-token log-probabilities, speaking rate, how many numbers and capitalised rare
words were heard, repetition. It is trained on dev renderings only (dev voices, dev sentences) and
its threshold is set on dev for a target recall; test numbers come from the held-out voices.

A chart error is any reference number lost or invented, any drug or nursing term lost, or the
resident's name lost.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .eval2 import OUT, fold, items, scoring_lexicon
from .metrics import score_utt

FEATURES = ["mean_lp", "min_lp", "p10_lp", "share_below_1", "share_below_3", "tokens_per_s", "chars_per_s",
            "numbers", "capital_words", "repeat_share", "n_tokens"]


def features(row: dict) -> np.ndarray:
    lp = np.array(row["logprobs"] or [-10.0])
    hyp, secs = row["hyp"], max(row["audio_seconds"], 0.1)
    words = hyp.split()
    caps = sum(1 for w in words[1:] if w[:1].isupper() and len(w) > 6)
    grams = Counter(zip(words, words[1:]))
    rep = sum(c - 1 for c in grams.values() if c > 1) / max(1, len(words))
    return np.array([lp.mean(), lp.min(), np.percentile(lp, 10), (lp < -1).mean(), (lp < -3).mean(),
                     len(lp) / secs, len(hyp) / secs, len(re.findall(r"\d", hyp)), caps, rep, len(lp)], dtype=np.float32)


def chart_error(row: dict, meta: dict, lex) -> bool:
    s = score_utt(row["ref"], row["hyp"], lex)
    if s.nums_hit < len(s.nums_ref) or s.nums_extra:
        return True
    if len(s.terms_hit) < len(s.terms_ref):
        return True
    res = meta.get(row["id"], {}).get("resident")
    return bool(res and fold(res) not in fold(row["hyp"]))


class Gate(torch.nn.Module):
    def __init__(self, n: int = len(FEATURES)):
        super().__init__()
        self.lin = torch.nn.Linear(n, 1)
        self.register_buffer("mu", torch.zeros(n))
        self.register_buffer("sd", torch.ones(n))
        self.threshold = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin((x - self.mu) / self.sd)).squeeze(-1)


def fit(x: np.ndarray, y: np.ndarray, target_recall: float = 0.8, epochs: int = 400, seed: int = 0) -> Gate:
    torch.manual_seed(seed)
    g = Gate(x.shape[1])
    X, Y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    g.mu.copy_(X.mean(0))
    g.sd.copy_(X.std(0).clamp_min(1e-3))
    opt = torch.optim.LBFGS(g.lin.parameters(), max_iter=epochs, line_search_fn="strong_wolfe")
    pos_w = (1 - Y.mean()) / Y.mean().clamp_min(1e-3)

    def closure():
        opt.zero_grad()
        p = g(X).clamp(1e-6, 1 - 1e-6)
        loss = -(pos_w * Y * p.log() + (1 - Y) * (1 - p).log()).mean() + 1e-3 * g.lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = g(X).numpy()
    # the lowest threshold-free choice: flag the fewest notes that still reach the target recall on dev
    order = np.sort(p[y == 1])
    g.threshold = float(order[int((1 - target_recall) * len(order))]) if len(order) else 0.5
    return g


def evaluate(decoder: str = "finetuned+bias", target_recall: float = 0.8, path: Path = OUT / "utterances.jsonl") -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows = [r for r in rows if r["decoder"] == decoder]
    meta = {it["id"]: it for ts in ("v1", "facility", "v1-dev", "facility-dev") for it in items(ts)}
    lex = scoring_lexicon()
    dev = [r for r in rows if r["set"] in ("v1-dev", "facility-dev")]
    test = [r for r in rows if r["set"] in ("v1", "facility")]
    if not dev or not test:
        raise ValueError(f"need dev and test rows for {decoder}")
    xd, yd = np.stack([features(r) for r in dev]), np.array([chart_error(r, meta, lex) for r in dev])
    xt, yt = np.stack([features(r) for r in test]), np.array([chart_error(r, meta, lex) for r in test])
    g = fit(xd, yd, target_recall)
    with torch.no_grad():
        pt = g(torch.from_numpy(xt)).numpy()
    flag = pt >= g.threshold
    res = {
        "decoder": decoder, "target_recall_on_dev": target_recall, "threshold": round(g.threshold, 4),
        "dev_notes": len(dev), "test_notes": len(test),
        "test_error_rate": round(float(yt.mean()), 4),
        "flag_rate": round(float(flag.mean()), 4),
        "recall": round(float(flag[yt].mean()) if yt.any() else 0.0, 4),
        "precision": round(float(yt[flag].mean()) if flag.any() else 0.0, 4),
        "silent_error_rate_without_gate": round(float(yt.mean()), 4),
        "silent_error_rate_with_gate": round(float((yt & ~flag).mean()), 4),
        "weights": dict(zip(FEATURES, [round(float(w), 3) for w in g.lin.weight[0].detach()])),
    }
    (OUT / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    torch.save({"state": g.state_dict(), "threshold": g.threshold, "features": FEATURES}, OUT / "gate.pt")
    return res
