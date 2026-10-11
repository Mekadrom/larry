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
    ("II", "or", None),  # fill in once "Schedule I or II" handling is decided
    ("VIII", "Henry", [["EIGHT"], ["THE", "EIGHTH"]]),
    ("X", "Brand", [["X"]]),
    ("I", "And", [["I"]]),
]


@pytest.mark.parametrize("token, prev, expected", CASES)
def test_normalize_token(token: str, prev: str | None, expected: list[list[str]] | None) -> None:
    if expected is None:
        pytest.skip("undecided")
    assert SCOTUSTextNormalizer().normalize_token(token, prev) == expected
