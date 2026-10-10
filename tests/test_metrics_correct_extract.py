import math

from care_stt.correct import LexiconCorrector, koelner_phonetik
from care_stt.extract import Extraction, extract, score_extraction
from care_stt.lexicon import Lexicon
from care_stt.metrics import aggregate, score_utt

LEX = Lexicon.load()


# ---------- lexicon & metrics ----------

def test_lexicon_matches_multiword_and_counts_occurrences():
    assert sorted(LEX.terms_in("braden skala erhoben puls 72 puls 80")) == ["Braden-Skala", "Puls", "Puls"]


def test_perfect_hypothesis_scores_perfectly():
    s = score_utt("Ramipril 5 Milligramm um acht Uhr gegeben.", "Ramipril fünf mg um 8 Uhr gegeben", LEX)
    a = aggregate([s])
    assert a.wer == 0 and a.term_recall == 1 and a.number_recall == 1 and a.number_utt_acc == 1


def test_number_metrics_catch_single_digit_error():
    s = score_utt("Blutdruck 130 zu 80, Puls 72.", "Blutdruck 133 zu 80, Puls 72.", LEX)
    a = aggregate([s])
    assert math.isclose(a.number_recall, 2 / 3)
    assert a.number_utt_acc == 0
    assert math.isclose(a.number_halluc, 1 / 3)


def test_term_recall_counts_missed_drug():
    a = aggregate([score_utt("Ramipril wie verordnet gegeben", "Ramibril wie verordnet gegeben", LEX)])
    assert a.term_recall == 0


# ---------- correction ----------

def test_koelner_phonetik_reference_values():
    # standard examples for the algorithm
    assert koelner_phonetik("Müller-Lüdenscheidt") == "65752682"
    assert koelner_phonetik("Wikipedia") == "3412"
    assert koelner_phonetik("Ramibril") == koelner_phonetik("Ramipril")


def test_corrects_misspelled_drug_and_keeps_punctuation():
    c = LexiconCorrector(LEX, 0.75)
    text, corrs = c.correct("Bewohnerin hat 5 MG Ramibril erhalten.")
    assert "Ramipril" in text and corrs[0].replacement == "Ramipril"
    text, _ = c.correct("Blutruck 130-80, Puls 72.")
    assert text.startswith("Blutdruck 130-80,")


def test_corrects_split_word():
    text, _ = LexiconCorrector(LEX, 0.75).correct("Pantopra zol vor dem Frühstück")
    assert text.startswith("Pantoprazol vor")


def test_does_not_touch_ordinary_words_or_numbers():
    sent = "Bewohner hat das Mittagessen vollständig gegessen, 2 Portionen."
    text, corrs = LexiconCorrector(LEX, 0.75).correct(sent)
    assert text == sent and corrs == []


def test_threshold_controls_aggressiveness():
    strict, loose = LexiconCorrector(LEX, 0.99), LexiconCorrector(LEX, 0.6)
    assert strict.correct("Ramnibryl gegeben")[1] == []
    assert loose.correct("Ramnibryl gegeben")[0].startswith("Ramipril")


# ---------- extraction ----------

def test_extracts_vitals_and_doses():
    ex = extract("Blutdruck 130 zu 80, Puls 72. Temperatur 37,8 Grad, Sauerstoffsättigung 94 Prozent.", LEX)
    assert ex.vitals == {"bp_sys": 130, "bp_dia": 80, "pulse": 72, "temp": 37.8, "spo2": 94}
    assert not ex.needs_review
    ex = extract("Ramipril 5 Milligramm um acht Uhr gegeben. 6 Einheiten Insulin.", LEX)
    assert ["Ramipril", 5, "mg"] in ex.doses and ["Insulin", 6, "IE"] in ex.doses


def test_flags_implausible_inconsistent_missing_unknown():
    assert ["pulse", "implausible"] in extract("Puls 22", LEX).flags
    assert ["bp", "inconsistent"] in extract("Blutdruck 80 zu 130", LEX).flags
    assert ["bp", "missing_value"] in extract("Blutdruck, Puls 72", LEX).flags
    assert ["dose", "unknown_drug"] in extract("Ramnibryl 5 Milligramm gegeben", LEX).flags


def test_blood_sugar_with_context_words():
    assert extract("Blutzucker vor dem Mittagessen 145, 4 Einheiten Insulin", LEX).vitals["glucose"] == 145


def test_score_extraction_separates_silent_from_flagged_errors():
    golds = [{"vitals": {"pulse": 72}}, {"vitals": {"pulse": 72}}, {"vitals": {"pulse": 72}}]
    preds = [Extraction(vitals={"pulse": 72}),
             Extraction(vitals={"pulse": 22}, flags=[["pulse", "implausible"]]),
             Extraction(vitals={"pulse": 77})]
    s = score_extraction(golds, preds)
    assert (s["correct"], s["wrong_flagged"], s["wrong_silent"]) == (1, 1, 1)
    assert s["wrong_caught_share"] == 0.5


def test_inflected_forms_are_not_rewritten():
    c = LexiconCorrector(LEX, 0.7)
    for sent in ("Bewohner war desorientiert.", "Leichte Ödemen an den Knöcheln."):
        assert c.correct(sent)[1] == []
