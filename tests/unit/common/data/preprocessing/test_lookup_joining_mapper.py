from pathlib import Path

import pytest
from datasets import Dataset

from larry.common.config.data.preprocessing.mapper_configs import LookupJoiningMapperConfig
from larry.common.data.preprocessing.mappers import LookupJoiningMapper
from tests.helpers.common import provenance, write_parquet_split


def _lookup(tmp_path: Path, ids: list[int]) -> str:
    write_parquet_split(tmp_path / "lookup", "train", {
        "image_id": ids,
        "captions": [[{"id": i, "caption": f"caption {i}"}] for i in ids],
        "license": [0] * len(ids),
    })
    return str(tmp_path / "lookup" / "train" / "*.parquet")


def _mapper(tmp_path: Path, ids: list[int], **overrides: object) -> LookupJoiningMapper:
    config = LookupJoiningMapperConfig(**{
        "key_column": "image_id",
        "lookup_columns": ["captions"],
        "lookup_path": "parquet",
        "lookup_data_files": _lookup(tmp_path, ids),
        "num_proc": 1,
        **overrides,
    })
    return LookupJoiningMapper(provenance(), config)


def test_rows_gain_the_lookup_columns_of_their_key_and_nulls_when_missing(tmp_path: Path) -> None:
    images = Dataset.from_dict({"image_id": [3, 1, 99], "license": [4, 7, 8]})

    joined = _mapper(tmp_path, [1, 2, 3]).preprocess_dataset(images)

    assert joined["image_id"] == [3, 1, 99]
    assert joined["license"] == [4, 7, 8]
    assert joined["captions"] == [[{"id": 3, "caption": "caption 3"}], [{"id": 1, "caption": "caption 1"}], None]


def test_duplicate_lookup_keys_are_rejected(tmp_path: Path) -> None:
    images = Dataset.from_dict({"image_id": [1]})

    with pytest.raises(ValueError, match="duplicate"):
        _mapper(tmp_path, [1, 1]).preprocess_dataset(images)
