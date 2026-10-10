"""Corpus loading and the cached audio set (voice x condition)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import degrade, synth
from .config import CACHE_DIR, ROOT


@dataclass
class Utterance:
    id: str
    text: str
    split: str
    kind: str
    vitals: dict | None = None
    doses: list | None = None
    time_h: int | None = None


def load_corpus(path: Path | str = ROOT / "data/corpus.jsonl", split: str | None = None) -> list[Utterance]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    utts = [Utterance(**r) for r in rows]
    return [u for u in utts if split is None or u.split == split]


def _clean_path(u: Utterance, voice: str) -> Path:
    return CACHE_DIR / "audio" / "clean" / voice / f"{u.id}.npy"


def clean_audio(u: Utterance, voice: str) -> np.ndarray:
    p = _clean_path(u, voice)
    if p.exists():
        return np.load(p)
    x = synth.synthesize(u.text, voice, synth.speaking_rate(u.id, voice, u.split))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, x)
    return x


def babble_sources(u: Utterance, pool: list[Utterance], k: int = 4) -> list[np.ndarray]:
    """k other talkers: dev-voice renderings of *different* dev sentences, so
    the interfering speech never contains the target sentence or speaker."""
    rng = np.random.default_rng(synth.seed_of(u.id, "babble"))
    others = [p for p in pool if p.id != u.id]
    picks = rng.choice(len(others), size=k, replace=False)
    voices = synth.DEV_VOICES
    return [clean_audio(others[i], voices[n % len(voices)]) for n, i in enumerate(picks)]


def audio(u: Utterance, voice: str, condition: str, pool: list[Utterance]) -> np.ndarray:
    if condition == "clean":
        return clean_audio(u, voice)
    p = CACHE_DIR / "audio" / condition / voice / f"{u.id}.npy"
    if p.exists():
        return np.load(p)
    x = clean_audio(u, voice)
    srcs = babble_sources(u, pool) if condition in ("babble_10", "babble_5", "combined") else []
    y = degrade.apply(condition, x, srcs, synth.seed_of(u.id, voice, condition))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, y)
    return y
