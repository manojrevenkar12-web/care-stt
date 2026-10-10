"""Lexicon-constrained post-correction.

Whisper has never seen most German drug names often enough to spell them:
"Ramipril" comes back as "Ramibril", "Ramnibryl" or "engeranibrile". A care
facility's medication plan is known in advance, so we can snap near-misses to
the closest lexicon entry - but only when the evidence is strong, because a
wrong "correction" (an ordinary word turned into a drug name) is worse than the
original error.

Candidate spans of 1-3 hypothesis words are compared to each lexicon term on
  * spelling: normalised Levenshtein similarity, and
  * sound:    similarity of Kölner Phonetik codes (a German phonetic hash, so
              "Ramibril" and "Ramipril" collide the way they sound alike).
A span is replaced when the combined score clears a threshold that is tuned on
the dev split (different sentences, different voices) and only then applied to
test. Numbers are never touched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import Lexicon

# Frequent German function/common words that must never be rewritten.
STOPWORDS = set("""
der die das den dem des ein eine einen einem einer und oder aber nicht nach vor mit
ohne von zu zum zur im in am an auf aus bei bis für über unter um wie wird wurde
werden hat haben ist sind war waren sich seit noch heute morgen abend nacht tag
tage stunden minuten uhr alle beim vom ins sehr gut wenig mehr weniger leicht
links rechts linken rechten keine kein beide beiden bewohner bewohnerin pflege
pflegekraft gegeben gespritzt verabreicht durchgeführt angeordnet gewechselt
""".split())


def koelner_phonetik(word: str) -> str:
    """Kölner Phonetik (Postel 1969) - standard phonetic code for German."""
    w = word.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "s")
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return ""
    codes = []
    for i, c in enumerate(w):
        prev = w[i - 1] if i > 0 else "#"   # sentinel: '' would match every `in` test
        nxt = w[i + 1] if i + 1 < len(w) else "#"
        if c in "aeijouy":
            code = "0"
        elif c == "h":
            code = ""
        elif c == "b":
            code = "1"
        elif c == "p":
            code = "3" if nxt == "h" else "1"
        elif c in "dt":
            code = "8" if nxt in "csz" else "2"
        elif c in "fvw":
            code = "3"
        elif c in "gkq":
            code = "4"
        elif c == "c":
            if i == 0:
                code = "4" if nxt in "ahkloqrux" else "8"
            else:
                code = "8" if prev in "sz" or nxt not in "ahkoqux" else "4"
        elif c == "x":
            code = "8" if prev in "ckq" else "48"
        elif c == "l":
            code = "5"
        elif c in "mn":
            code = "6"
        elif c == "r":
            code = "7"
        elif c in "sz":
            code = "8"
        else:
            code = ""
        codes.append(code)
    out = []
    for code in "".join(codes):
        if not out or out[-1] != code:
            out.append(code)
    s = "".join(out)
    return s[0] + s[1:].replace("0", "") if s else s


# German inflection endings. A span that differs from a term only by these is
# another form of the same word ("desorientiert" vs "Desorientierung",
# "Ödemen" vs "Ödeme"), not a misrecognition, and is left alone.
_INFLECTIONS = {"", "e", "en", "n", "s", "es", "er", "ern", "t", "et", "te", "ten", "ung", "ungen"}


def is_inflection(span: str, term: str) -> bool:
    i = 0
    while i < min(len(span), len(term)) and span[i] == term[i]:
        i += 1
    return i >= len(term) - 4 and span[i:] in _INFLECTIONS and term[i:] in _INFLECTIONS


def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - _lev(a, b) / max(len(a), len(b))


@dataclass
class Correction:
    original: str
    replacement: str
    score: float


_WORD = re.compile(r"^([A-Za-zÄÖÜäöüß\-]+)([.,;:!?]*)$")


class LexiconCorrector:
    def __init__(self, lexicon: Lexicon, threshold: float = 0.75, max_span: int = 3, min_len: int = 5):
        self.lexicon = lexicon
        self.threshold = threshold
        self.max_span = max_span
        self.min_len = min_len
        self._entries = []
        for term in lexicon.canonical:
            key = term.lower().replace("-", "")
            self._entries.append((term, key, koelner_phonetik(key)))
        self._exact = {k for _, k, _ in self._entries}

    def score(self, span: str, key: str, code: str) -> float:
        s = span.lower().replace("-", "")
        return 0.5 * similarity(s, key) + 0.5 * similarity(koelner_phonetik(s), code)

    def best(self, span: str) -> tuple[str, float] | None:
        s = span.lower().replace("-", "")
        if s in self._exact or len(s) < self.min_len:
            return None
        best = max(((t, self.score(span, k, c)) for t, k, c in self._entries
                    if 0.6 <= len(s) / len(k) <= 1.6), key=lambda x: x[1], default=None)
        if best and is_inflection(s, best[0].lower().replace("-", "")):
            return None
        return best

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        toks = text.split()
        parsed = [_WORD.match(t) for t in toks]
        candidates = []  # (score, start, end, term)
        for i in range(len(toks)):
            for n in range(1, self.max_span + 1):
                j = i + n
                if j > len(toks):
                    break
                ms = parsed[i:j]
                if not all(ms):
                    break
                words = [m.group(1) for m in ms]
                if any(w.lower() in STOPWORDS for w in words):
                    break
                # inner tokens of a multi-word span must not carry punctuation
                if any(m.group(2) for m in ms[:-1]):
                    break
                joined = "".join(words)
                if n > 1 and joined.lower().replace("-", "") in self._exact:
                    # "Pantopra zol" -> exact term once re-joined
                    term = next(t for t, k, _ in self._entries if k == joined.lower().replace("-", ""))
                    candidates.append((1.0, i, j, term))
                    continue
                hit = self.best(joined)
                if hit and hit[1] >= self.threshold:
                    candidates.append((hit[1], i, j, hit[0]))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        taken = [False] * len(toks)
        chosen = []
        for sc, i, j, term in candidates:
            if not any(taken[i:j]):
                chosen.append((i, j, term, sc))
                for k in range(i, j):
                    taken[k] = True
        corrections = []
        out = list(toks)
        for i, j, term, sc in sorted(chosen, reverse=True):
            punct = parsed[j - 1].group(2)
            corrections.append(Correction(" ".join(toks[i:j]), term, round(sc, 3)))
            out[i:j] = [term + punct]
        return " ".join(out), corrections[::-1]
