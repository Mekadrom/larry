from datasets import Dataset

from larry.text.config.preprocessing.text_mapper_configs import CaptionSelectingMapperConfig
from larry.text.data.preprocessing.text_mappers import CaptionSelectingMapper
from tests.helpers.common import provenance


def _select(rows: dict[str, list[object]], columns: list[str], **overrides: object) -> Dataset:
    config = CaptionSelectingMapperConfig(caption_columns=columns, **overrides)
    return CaptionSelectingMapper(provenance(), config).preprocess_dataset(Dataset.from_dict(rows))


def test_the_first_usable_column_in_priority_order_wins() -> None:
    mapped = _select({
        "visual": ["A red vase.", None, "  "],
        "title": ["Vase", "Bowl", "Cup"],
    }, ["visual", "title"])

    assert mapped["caption"] == ["A red vase.", "Bowl", "Cup"]
    assert mapped["caption_source"] == ["visual", "title", "title"]


def test_list_columns_use_their_first_non_empty_entry() -> None:
    mapped = _select({"captions": [["", "A dog on grass.", "A puppy."], []]}, ["captions"])

    assert mapped["caption"] == ["A dog on grass.", None]


def test_short_text_is_skipped_and_captionless_rows_can_be_filtered() -> None:
    mapped = _select(
        {"visual": ["Hat", None], "title": ["Unidentified Man", "Tiny"]},
        ["visual", "title"],
        min_characters=5,
        filter_null_outputs=True,
    )

    assert mapped["caption"] == ["Unidentified Man"]
    assert mapped["caption_source"] == ["title"]
