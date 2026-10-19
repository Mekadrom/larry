import json
from collections import Counter
from typing import Any

import pytest
from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.preprocessor_configs import (
    JsonPathMapping,
    NestedExtractionPreprocessorConfig,
)
from larry.common.data.preprocessing.preprocessors import NestedExtractionPreprocessor
from tests.helpers.common import provenance


def _exif_dataset(exif: dict[str, Any]) -> Dataset:
    return Dataset.from_dict({"exif": [json.dumps(exif)]})


_config = NestedExtractionPreprocessorConfig(
    input_column="exif",
    json_path_mappings=[
        JsonPathMapping(json_path="$.'Image Copyright'", output_column="exif_copyright"),
        JsonPathMapping(json_path="$.'Image XPComment'", output_column="text_original"),
    ],
)


@pytest.mark.parametrize(
    ("expected_text_original", "expected_copyright", "dataset", "config"),
    [
        (
            "Photo depicts bruh",
            "CC-BY-4.0",
            _exif_dataset({"Image Copyright": "CC-BY-4.0", "Image XPComment": "Photo depicts bruh"}),
            _config,
        ),
        ("Photo depicts hurb", None, _exif_dataset({"Image XPComment": "Photo depicts hurb"}), _config),
        (None, None, _exif_dataset({"Unrelated Field": "bruh"}), _config),
    ]
)
def test_nested_extraction_extracts(
        expected_text_original: str | None,
        expected_copyright: str | None,
        dataset: Dataset | DatasetDict,
        config: NestedExtractionPreprocessorConfig
) -> None:
    sut = NestedExtractionPreprocessor(provenance(), config)

    dataset = sut.preprocess_dataset(dataset)

    assert Counter(dataset["text_original"]) == Counter([expected_text_original])
    assert Counter(dataset["exif_copyright"]) == Counter([expected_copyright])
