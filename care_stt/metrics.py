"""Scoring. Word error rate alone hides the errors that matter in care
documentation: one wrong digit in a dosage is a single substitution, the same
cost as "der" -> "die". So alongside WER/CER we report:

  term_recall     - share of lexicon terms (drugs, nursing terms) in the
                    reference that appear verbatim in the hypothesis
  number_recall   - share of numbers in the reference (vitals, doses, times)
                    reproduced exactly
  number_utt_acc  - share of utterances whose numbers are ALL exactly right
  number_halluc   - numbers in hypotheses that are not in the reference,
                    per reference number
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import jiwer

from .lexicon import Lexicon
from .normalize import normalize, numbers_in


@dataclass
class UttScore:
    ref: str
    hyp: str
    terms_ref: list[str]
    terms_hit: list[str]
    nums_ref: list[str]
    nums_hyp: list[str]

    @property
    def nums_hit(self) -> int:
        return sum((Counter(self.nums_ref) & Counter(self.nums_hyp)).values())

    @property
    def nums_extra(self) -> int:
        return sum((Counter(self.nums_hyp) - Counter(self.nums_ref)).values())


def score_utt(ref_text: str, hyp_text: str, lexicon: Lexicon) -> UttScore:
    ref, hyp = normalize(ref_text), normalize(hyp_text)
    terms = lexicon.terms_in(ref)
    hyp_terms = set(lexicon.terms_in(hyp))
    return UttScore(ref, hyp, terms, [t for t in terms if t in hyp_terms], numbers_in(ref), numbers_in(hyp))


@dataclass
class Aggregate:
    n: int
    wer: float
    cer: float
    term_recall: float
    n_terms: int
    number_recall: float
    number_utt_acc: float
    number_halluc: float
    n_numbers: int
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def aggregate(scores: list[UttScore]) -> Aggregate:
    refs = [s.ref for s in scores]
    hyps = [s.hyp if s.hyp else "<empty>" for s in scores]
    n_terms = sum(len(s.terms_ref) for s in scores)
    n_nums = sum(len(s.nums_ref) for s in scores)
    num_utts = [s for s in scores if s.nums_ref]
    return Aggregate(
        n=len(scores),
        wer=jiwer.wer(refs, hyps),
        cer=jiwer.cer(refs, hyps),
        term_recall=sum(len(s.terms_hit) for s in scores) / n_terms if n_terms else float("nan"),
        n_terms=n_terms,
        number_recall=sum(s.nums_hit for s in scores) / n_nums if n_nums else float("nan"),
        number_utt_acc=(sum(s.nums_hit == len(s.nums_ref) and s.nums_extra == 0 for s in num_utts) / len(num_utts)
                        if num_utts else float("nan")),
        number_halluc=sum(s.nums_extra for s in scores) / n_nums if n_nums else float("nan"),
        n_numbers=n_nums,
    )
