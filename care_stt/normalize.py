"""German text normalisation for ASR scoring.

Whisper writes "130 zu 80", "130/80", "130-80" or "hundertdreißig zu achtzig"
for the same dictation. Scoring raw strings would count all but one of those as
errors, so reference and hypothesis go through the same normaliser:

  * lowercase, ß -> ss, punctuation removed (decimal commas between digits kept)
  * German number words -> digits ("siebenunddreißig komma acht" -> "37,8",
    "zweieinhalb" -> "2,5"); the bare article "ein/eine" is left alone
  * digit-by-digit compounds "9 und 60" -> "69" (Whisper's rendering of
    "neunundsechzig")
  * blood-pressure style "130/80" and "130-80" -> "130 zu 80"
  * units: mg -> milligramm, ml -> milliliter, % -> prozent, °/°c -> grad
  * hyphens between words become spaces

The rules are applied identically to both sides, so they cannot hide a real
recognition error - they only remove formatting differences.
"""
from __future__ import annotations

import re

_UNITS = {
    "null": 0, "eins": 1, "ein": 1, "eine": 1, "zwei": 2, "zwo": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11,
    "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
    "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
}
_TENS = {"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50,
         "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90}
# Standalone words that are articles far more often than numbers.
_NOT_STANDALONE = {"ein", "eine", "einen", "einem", "einer"}
_UNIT_WORDS = {"mg": "milligramm", "ml": "milliliter", "%": "prozent", "°c": "grad",
               "°": "grad", "ie": "einheiten", "i.e.": "einheiten"}


def parse_number_word(word: str) -> int | None:
    """Parse one German number word (compounds allowed) to an int, else None."""
    w = word.lower()
    if not w or not re.fullmatch(r"[a-zäöüß]+", w):
        return None
    return _parse(w)


def _parse(w: str) -> int | None:
    if w == "":
        return 0
    if w in _UNITS:
        return _UNITS[w]
    if w in _TENS:
        return _TENS[w]
    for word, mult in (("tausend", 1000), ("hundert", 100)):
        if word in w:
            left, right = w.split(word, 1)
            lv = 1 if left in ("", "ein", "eins") else _parse(left)
            rv = _parse(right) if right else 0
            if lv is None or rv is None or lv == 0 or (rv >= mult):
                return None
            return lv * mult + rv
    if "und" in w:
        left, right = w.split("und", 1)
        lv = 1 if left == "ein" else _UNITS.get(left)
        rv = _TENS.get(right)
        if lv is not None and rv is not None and 0 < lv < 10:
            return lv + rv
    return None


def _half_compound(w: str) -> str | None:
    """'zweieinhalb' -> '2,5', 'eineinhalb' -> '1,5'."""
    if w.endswith("einhalb") and len(w) > len("einhalb"):
        base = w[: -len("einhalb")]
        v = 1 if base == "ein" else parse_number_word(base)
        if v is not None:
            return f"{v},5"
    return None


def normalize(text: str) -> str:
    t = text.lower().replace("ß", "ss")
    # numeric formatting before punctuation is stripped
    t = re.sub(r"(\d)\.(\d)", r"\1,\2", t)                 # 37.8 -> 37,8
    t = re.sub(r"(\d{2,3})\s*[/\-]\s*(\d{2,3})", r"\1 zu \2", t)  # 130/80, 130-80
    t = re.sub(r"(\d)\s*%", r"\1 %", t)
    t = re.sub(r"(\d)\s*°\s*c?\b", r"\1 °", t)
    t = re.sub(r"(\d)(mg|ml)\b", r"\1 \2", t)
    t = re.sub(r"(?<=[a-zäöü])-(?=[a-zäöü])", " ", t)   # Braden-Skala -> braden skala
    # tokenise: keep decimal numbers, words, % and °
    tokens = re.findall(r"\d+(?:,\d+)?|[a-zäöü]+|%|°", t)
    out: list[str] = []
    for tok in tokens:
        tok = _UNIT_WORDS.get(tok, tok)
        if tok.isalpha():
            half = _half_compound(tok)
            if half:
                out.append(half)
                continue
            if tok not in _NOT_STANDALONE:
                v = parse_number_word(tok)
                if v is not None:
                    out.append(str(v))
                    continue
        out.append(tok)
    # Whisper often writes a German compound number digit by digit:
    # "neunundsechzig" (69) comes back as "9 und 60". Recompose units + tens.
    out = _recompose_und(out)
    # "37 komma 8" -> "37,8"
    merged: list[str] = []
    i = 0
    while i < len(out):
        if (i + 2 < len(out) and out[i].isdigit() and out[i + 1] == "komma" and out[i + 2].isdigit()):
            merged.append(f"{out[i]},{out[i + 2]}")
            i += 3
        else:
            merged.append(out[i])
            i += 1
    return " ".join(merged)


def _recompose_und(tokens: list[str]) -> list[str]:
    res: list[str] = []
    i = 0
    while i < len(tokens):
        if (i + 2 < len(tokens) and tokens[i].isdigit() and tokens[i + 1] == "und" and tokens[i + 2].isdigit()
                and 1 <= int(tokens[i]) <= 9 and int(tokens[i + 2]) in range(20, 100, 10)):
            res.append(str(int(tokens[i]) + int(tokens[i + 2])))
            i += 3
        else:
            res.append(tokens[i])
            i += 1
    return res


_NUM_RE = re.compile(r"^\d+(?:,\d+)?$")


def numbers_in(normalized: str) -> list[str]:
    """Numeric tokens of an already-normalised string, in order."""
    return [t for t in normalized.split() if _NUM_RE.match(t)]
