import pytest

from larry.voice.data.preprocessing.normalization import SCOTUSTextNormalizer

CASES = [
    ("Mr.", None, [["MISTER"]]),
    ("MR.", None, [["MISTER"]]),
    ("Mrs.", None, [["MISSUS"]]),
    ("v.", None, [["VERSUS"], ["V"]]),
    ("(c)(1)(v)", None, [["C", "ONE", "V"]]),
    ("III's", "Article", [["THREE'S"]]),
    ("II(2)", "Article", [["TWO", "TWO"]]),
    ("VIII", "Henry", [["EIGHT"], ["THE", "EIGHTH"]]),
    ("X", "Brand", [["X"]]),
    ("I", "And", [["I"]]),
]


@pytest.mark.parametrize("token, prev, expected", CASES)
def test_normalize_token(token: str, prev: str | None, expected: list[list[str]]) -> None:
    assert SCOTUSTextNormalizer().normalize_token(token, prev) == expected


SEQUENCES = [
    ("Schedule I or II drugs", [[["ONE"]], [["OR"]], [["TWO"]], [["DRUGS"]]]),
    ("Schedule I, II, or III", [[["ONE"]], [["TWO"]], [["OR"]], [["THREE"]]]),
    ("Schedule I, II; and IV", [[["ONE"]], [["TWO"]], [["AND"]], [["FOUR"]]]),
    ("World War I and II", [[["ONE"]], [["AND"]], [["TWO"]]]),
    ("Henry VIII and IX", [[["EIGHT"], ["THE", "EIGHTH"]], [["AND"]], [["NINE"], ["THE", "NINTH"]]]),
    ("Article III and I think", [[["THREE"]], [["AND"]], [["I"], ["ONE"]], [["THINK"]]]),
    ("you and I", [[["AND"]], [["I"]]]),
    ("Title VII, and the", [[["SEVEN"]], [["AND"]], [["THE"]]]),
]


@pytest.mark.parametrize("sentence, expected", SEQUENCES)
def test_roman_numerals_carry_their_context_across_lists(sentence: str, expected: list[list[list[str]]]) -> None:
    tokens = sentence.split()
    normalizer = SCOTUSTextNormalizer()

    readings = [normalizer.normalize_in_context(tokens, i) for i in range(len(tokens))]

    assert readings[-len(expected):] == expected
