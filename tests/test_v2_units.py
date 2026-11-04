"""v2 units without model weights: the data split, the bias tree, the fold used to score names."""
from care_stt import corpus2
from care_stt.eval2 import fold
from care_stt.runtime import BiasTree


def test_training_text_never_contains_test_residents_or_held_out_drugs():
    train = " ".join(r["text"] for r in corpus2.train_corpus())
    for f in {**corpus2.FACILITIES, **corpus2.DEV_FACILITY}.values():
        for name in f["residents"]:
            assert name not in train, name
    for d in corpus2.HELDOUT_DRUGS + corpus2.NEW_DRUGS:
        assert d not in train, d
    assert not set(corpus2.NEW_DRUGS) & set(corpus2.LEXICON_DRUGS)


def test_facility_test_covers_every_drug_class_and_is_reproducible():
    a, b = corpus2.facility_test(), corpus2.facility_test()
    assert a == b
    classes = {r["drug_class"] for r in a}
    assert classes == {"train", "heldout", "new"}
    for r in a:
        assert r["resident"] in corpus2.FACILITIES[r["facility"]]["residents"]
        assert r["drug"] in corpus2.FACILITIES[r["facility"]]["plan"]
        assert r["resident"] in r["text"] and r["drug"] in r["text"]


def test_dev_facility_shares_nothing_with_test_facilities():
    dev = corpus2.DEV_FACILITY["Haus Sonnenhang"]
    test = [x for f in corpus2.FACILITIES.values() for x in f["residents"] + f["plan"]]
    assert not set(dev["residents"]) & set(test)
    assert not (set(dev["plan"]) - set(corpus2.TRAIN_DRUGS)) & set(test)


def test_v1_test_notes_share_no_six_word_run_with_training_text():
    import json

    from care_stt.config import ROOT

    def grams(s, n=6):
        w = s.lower().replace(",", " ").replace(".", " ").split()
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

    train = set().union(*(grams(r["text"]) for r in corpus2.train_corpus()))
    v1 = [json.loads(line) for line in (ROOT / "data/corpus.jsonl").read_text().splitlines()]
    assert not [r for r in v1 if r["split"] == "test" and r["kind"] == "narrative" and grams(r["text"]) & train]


class StubTok:
    """Each character is a token; enough to exercise the prefix tree."""

    eot = 10_000

    def __init__(self):
        self.pieces, self.ids = {}, {}

    def _id(self, ch: bytes) -> int:
        if ch not in self.ids:
            i = len(self.ids)
            self.ids[ch], self.pieces[i] = i, ch
        return self.ids[ch]

    def encode(self, s: str) -> list[int]:
        return [self._id(c.encode()) for c in (" " + s.strip())]


def test_bias_tree_rewards_completed_phrases_and_takes_back_broken_ones():
    tok = StubTok()
    tree = BiasTree(tok, ["Abc"])
    ids = tok.encode("Abc")
    active, earned, completed = [], 0, None
    for t in ids:
        active, gained, lost, c = tree.step(active, t)
        earned += gained - lost
        completed = completed or c
    assert completed == "Abc" and earned == len(ids) and active == []
    # a partial match that breaks off gives back what it earned
    active, earned = [], 0
    for t in tok.encode("Abx")[:-1] + [tok._id(b"x")]:
        active, gained, lost, _ = tree.step(active, t)
        earned += gained - lost
    assert earned == 0
    assert tok.encode("Abc")[0] in tree.boosts([])


def test_fold_matches_names_across_diacritics():
    assert fold("Erdoğan") == fold("Erdogan")
    assert fold("Szymańska") == fold("SZYMANSKA")
    assert fold("Weißenborn") != fold("Weissenborn")  # ß is a letter, not a diacritic
