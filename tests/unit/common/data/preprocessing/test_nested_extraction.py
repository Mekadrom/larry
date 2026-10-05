import json
from collections import Counter

import pytest
from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.preprocessor_configs import NestedExtractionPreprocessorConfig
from larry.common.data.preprocessing.preprocessors import NestedExtractionPreprocessor


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


_config = NestedExtractionPreprocessorConfig(
    input_column="exif",
    json_path_to_output_column_mappings=[
        {"jsonPath": "$.'Image Copyright'", "outputColumn": "exif_copyright"},
        {"jsonPath": "$.'Image XPComment'", "outputColumn": "text_original"},
    ],
)


@pytest.mark.parametrize(
    ("expected_text_original", "expected_copyright", "dataset", "config"),
    [
        ("Photo depicts bruh", "CC-BY-4.0", dataset_both(), _config),
        ("Photo depicts hurb", None, dataset_one(), _config),
        (None, None, dataset_neither(), _config),
    ]
)
def test_nested_extraction_extracts(
        expected_text_original: str | None,
        expected_copyright: str | None,
        dataset: Dataset | DatasetDict,
        config: NestedExtractionPreprocessorConfig
) -> None:
    sut = NestedExtractionPreprocessor(config)

    dataset = sut.preprocess_dataset(dataset)

    assert Counter(dataset["text_original"]) == Counter([expected_text_original])
    assert Counter(dataset["exif_copyright"]) == Counter([expected_copyright])
