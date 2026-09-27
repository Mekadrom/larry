import json
from collections import Counter

import pytest
from datasets import Dataset

from larry.common.config.preprocessing_mapper_configs import NestedExtractionPreprocessingMapperConfig
from larry.common.data.preprocessing.preprocessing_mappers import NestedExtractionPreprocessingMapper


def dataset_both() -> Dataset:
    exifs = [
        {
            "Image Copyright": "CC-BY-4.0",
            "Image XPComment": "Photo depicts bruh"
        }
    ]
    return Dataset.from_dict({"exif": [json.dumps(e) for e in exifs]})


def dataset_one() -> Dataset:
    exifs = [
        {
            "Image XPComment": "Photo depicts hurb"
        }
    ]
    return Dataset.from_dict({"exif": [json.dumps(e) for e in exifs]})


def dataset_neither() -> Dataset:
    exifs = [
        {
            "Unrelated Field": "bruh",
        },
    ]
    return Dataset.from_dict({"exif": [json.dumps(e) for e in exifs]})


_config_drop_any = NestedExtractionPreprocessingMapperConfig(
    input_column="exif",
    json_path_to_output_column_mappings=[
        {"jsonPath": "$.'Image Copyright'", "outputColumn": "exif_copyright"},
        {"jsonPath": "$.'Image XPComment'", "outputColumn": "text_original"},
    ],
    drop_missing_type="any",
)
_config_drop_all = NestedExtractionPreprocessingMapperConfig(
    input_column="exif",
    json_path_to_output_column_mappings=[
        {"jsonPath": "$.'Image Copyright'", "outputColumn": "exif_copyright"},
        {"jsonPath": "$.'Image XPComment'", "outputColumn": "text_original"},
    ],
    drop_missing_type="all",
)
_config_drop_never = NestedExtractionPreprocessingMapperConfig(
    input_column="exif",
    json_path_to_output_column_mappings=[
        {"jsonPath": "$.'Image Copyright'", "outputColumn": "exif_copyright"},
        {"jsonPath": "$.'Image XPComment'", "outputColumn": "text_original"},
    ],
    drop_missing_type="never",
)


@pytest.mark.parametrize(
    ("dataset", "config"),
    [
        (dataset_neither(), _config_drop_all),
        (dataset_neither(), _config_drop_any),
        (dataset_one(), _config_drop_any),
    ]
)
def test_nested_extraction_drops(dataset: Dataset, config: NestedExtractionPreprocessingMapperConfig) -> None:
    sut = NestedExtractionPreprocessingMapper(config)

    dataset = sut.preprocess_dataset(dataset)

    assert Counter(dataset["text_original"]) == Counter([])
    assert Counter(dataset["exif_copyright"]) == Counter([])

@pytest.mark.parametrize(
    ("expected_text_original", "expected_copyright", "dataset", "config"),
    [
        ("Photo depicts bruh", "CC-BY-4.0", dataset_both(), _config_drop_all),
        ("Photo depicts bruh", "CC-BY-4.0", dataset_both(), _config_drop_any),
        ("Photo depicts hurb", None, dataset_one(), _config_drop_all),
        (None, None, dataset_neither(), _config_drop_never),
    ]
)
def test_nested_extraction_extracts(
        expected_text_original: str | None,
        expected_copyright: str | None,
        dataset: Dataset,
        config: NestedExtractionPreprocessingMapperConfig
) -> None:
    sut = NestedExtractionPreprocessingMapper(config)

    dataset = sut.preprocess_dataset(dataset)

    assert Counter(dataset["text_original"]) == Counter([expected_text_original])
    assert Counter(dataset["exif_copyright"]) == Counter([expected_copyright])