from __future__ import annotations

from pathlib import Path

from .config import ROOT
from .normalize import normalize


class Lexicon:
    """Domain terms (drug names, nursing terms) in canonical spelling, plus
    their normalised token sequences for matching against transcripts."""

    def __init__(self, terms: list[str]):
        self.canonical = list(dict.fromkeys(terms))
        self.norm = {t: tuple(normalize(t).split()) for t in self.canonical}

    @classmethod
    def load(cls, path: Path | str = ROOT / "data/lexicon.txt") -> "Lexicon":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([l.strip() for l in lines if l.strip() and not l.startswith("#")])

    def terms_in(self, normalized_text: str) -> list[str]:
        """All lexicon terms occurring in a normalised string (one entry per
        occurrence; a term nested in a longer matched term is not double-counted)."""
        toks = normalized_text.split()
        found: list[str] = []
        used = [False] * len(toks)
        for term, seq in sorted(self.norm.items(), key=lambda kv: -len(kv[1])):
            n = len(seq)
            for i in range(len(toks) - n + 1):
                if tuple(toks[i:i + n]) == seq and not any(used[i:i + n]):
                    found.append(term)
                    for j in range(i, i + n):
                        used[j] = True
        return found
