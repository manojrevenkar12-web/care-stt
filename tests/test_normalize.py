import pytest

from care_stt.normalize import normalize, numbers_in, parse_number_word


@pytest.mark.parametrize("word,value", [
    ("sieben", 7), ("zwölf", 12), ("zwanzig", 20), ("siebenunddreißig", 37),
    ("einundzwanzig", 21), ("hundert", 100), ("einhundertdreißig", 130),
    ("hundertfünfundvierzig", 145), ("zweihundert", 200), ("zweitausendsechsundzwanzig", 2026),
    ("dreissig", 30),
])
def test_number_words(word, value):
    assert parse_number_word(word) == value


@pytest.mark.parametrize("word", ["bewohner", "und", "hunderte", "puls", "elfmeter", ""])
def test_non_numbers(word):
    assert parse_number_word(word) is None


@pytest.mark.parametrize("a,b", [
    ("Blutdruck 130 zu 80, Puls 72.", "blutdruck einhundertdreißig zu achtzig puls zweiundsiebzig"),
    ("Blutdruck 130/80", "Blutdruck 130 zu 80"),
    ("Blutdruck 130-80", "Blutdruck 130 zu 80"),
    ("Temperatur 37,8 Grad", "Temperatur siebenunddreißig Komma acht Grad"),
    ("Temperatur 37,8 Grad", "Temperatur 37.8°C"),
    ("2,5 Milligramm", "zweieinhalb mg"),
    ("Sättigung 94 Prozent", "Sättigung 94%"),
    ("Braden-Skala", "Braden Skala"),
    ("Fußpflege", "fusspflege"),
    ("Blutdruck 139 zu 69", "Blutdruck 139 zu 9 und 60"),
    ("Puls 53", "Puls drei und fünfzig"),
])
def test_equivalent_forms(a, b):
    assert normalize(a) == normalize(b)


def test_article_ein_is_not_a_number():
    assert normalize("ein Bewohner und eine Bewohnerin") == "ein bewohner und eine bewohnerin"


def test_real_error_is_not_hidden():
    # normalisation must not make a wrong reading look right
    assert normalize("Puls 72") != normalize("22 und 70")
    assert normalize("Blutdruck 130 zu 80") != normalize("Blutdruck 133 zu 80")


def test_numbers_in():
    assert numbers_in(normalize("Blutdruck 130 zu 80, Temperatur 37,8")) == ["130", "80", "37,8"]


def test_und_recomposition_is_narrow():
    # only units(1-9) + "und" + round tens; other "und" stays
    assert normalize("130 und 80") == "130 und 80"
    assert normalize("2 und 3 Tabletten") == "2 und 3 tabletten"
