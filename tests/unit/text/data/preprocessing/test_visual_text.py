import pytest
from datasets import Dataset

from larry.text.config.preprocessing.text_mapper_configs import VisualDescriptionExtractingMapperConfig
from larry.text.data.preprocessing.text_mappers import VisualDescriptionExtractingMapper
from larry.text.data.preprocessing.visual_text import VisualTextExtractor
from tests.helpers.common import provenance

VISUAL = (
    "Two delicately folded origami kimonos are presented to the viewer in a dark teal window mat and frame. "
    "The kimono on the left was made with a blue and white decorative paper featuring white and blue pine needles "
    "and blossoms."
)
HISTORY = (
    "After moving to Washington, DC following World War II, Ira Blount mastered dozens of artisanal crafts. "
    "His love for crafting, learning, and community engagement would endure over the course of his long life and "
    "career."
)


def _extractor(**overrides: object) -> VisualTextExtractor:
    return VisualTextExtractor(VisualDescriptionExtractingMapperConfig(**overrides))


def test_history_paragraphs_are_dropped_and_visual_ones_kept() -> None:
    text, retained = _extractor().extract(f"{VISUAL}\n\n{HISTORY}")

    assert text == VISUAL
    assert retained == pytest.approx(len(VISUAL) / len(f"{VISUAL}\n\n{HISTORY}"))


def test_a_historical_sentence_inside_a_visual_paragraph_is_dropped() -> None:
    text, _ = _extractor().extract(f"{VISUAL} The kimonos were donated to the museum in 1998 by his family.")

    assert text == VISUAL


def test_a_strongly_visual_sentence_survives_a_historical_paragraph() -> None:
    sentence = "The round blue porcelain vase is painted with white floral patterns on its curved surface."
    text, _ = _extractor().extract(f"{HISTORY} {sentence}")

    assert text == sentence


def test_text_without_any_visual_content_gives_nothing() -> None:
    assert _extractor().extract(HISTORY) == (None, 0.0)
    assert _extractor().extract(None) == (None, None)


@pytest.mark.parametrize(("paragraph", "expected"), [
    ("Made by Mrs. J. H. Allen in the U.S. Army. It is red.", [
        "Made by Mrs. J. H. Allen in the U.S. Army.", "It is red.",
    ]),
    ('He said "It is done." Then he left!', ['He said "It is done."', "Then he left!"]),
    ("Dated ca. 1890. Painted blue.", ["Dated ca. 1890.", "Painted blue."]),
])
def test_sentences_split_without_breaking_on_abbreviations_or_initials(paragraph: str, expected: list[str]) -> None:
    assert VisualTextExtractor.sentences(paragraph) == expected


def test_extra_terms_shift_the_decision() -> None:
    sentence = "A kimono with pine needles."
    default = _extractor(sentence_keep_threshold=0.1).extract(f"{HISTORY} {sentence}")[0]
    extended = _extractor(sentence_keep_threshold=0.1, extra_visual_terms=["kimono", "pine", "needles"])

    assert default is None
    assert extended.extract(f"{HISTORY} {sentence}")[0] == sentence


def test_the_mapper_adds_the_visual_text_and_how_much_was_kept() -> None:
    config = VisualDescriptionExtractingMapperConfig()
    dataset = Dataset.from_dict({"description": [f"{VISUAL}\n\n{HISTORY}", None]})

    mapped = VisualDescriptionExtractingMapper(provenance(), config).preprocess_dataset(dataset)

    assert mapped["visual_description"] == [VISUAL, None]
    assert mapped["visual_retained"][1] is None
    assert mapped["description"][0].endswith("career.")
