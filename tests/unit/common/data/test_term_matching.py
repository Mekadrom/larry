import pytest

from larry.common.data.term_matching import TermMatcher


@pytest.mark.parametrize(("text", "expected"), [
    ("Maori terraces", True),
    ("Group at the MARAE", True),
    ("Waikanae beach", False),
    ("Papakura station", False),
    ("A carved waka.", True),
    ("Pā site", True),
    ("Pātea", False),
    ("Image © Getty", True),
    ("plain text", False),
])
def test_terms_match_whole_words_regardless_of_case(text: str, expected: bool) -> None:
    matcher = TermMatcher(["Māori", "Maori", "marae", "waka", "pā", "©"])

    assert matcher.matches([text]) is expected


def test_any_of_several_texts_can_match_and_missing_texts_are_skipped() -> None:
    matcher = TermMatcher(["logo"])

    assert matcher.matches([None, "", "The NASA logo"])
    assert not matcher.matches([None, "launch pad"])


def test_composed_and_decomposed_macrons_match_the_same() -> None:
    matcher = TermMatcher(["pā"])

    assert matcher.matches(["pā site"])


def test_no_terms_never_match() -> None:
    assert not TermMatcher([]).matches(["anything"])
