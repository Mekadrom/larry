from pathlib import Path

import pytest

from larry.common.config.data.preprocessing.dataset_configs import IndexedSourceDatasetConfig
from larry.common.data.dataset_providers import DatasetProvider, IndexedSourceDatasetProvider
from tests.helpers.common import write_parquet_split


def _source(root: Path) -> Path:
    return write_parquet_split(root / "source", "train", {
        "id": ["a", "b", "c", "d"],
        "audio": [b"0", b"1", b"2", b"3"],
        "text": ["zero", "one", "two", "three"],
    })


def _index(root: Path, splits: dict[str, dict[str, list]]) -> Path:
    for split, columns in splits.items():
        write_parquet_split(root / "index", split, columns)
    return root / "index"


def _provider(index: Path, source: Path | None, split_sources: dict[str, str]) -> IndexedSourceDatasetProvider:
    config = IndexedSourceDatasetConfig(
        path=str(index), split_sources=split_sources, source_path=str(source) if source else None,
    )
    provider = DatasetProvider.REGISTRY.for_config(config)(config)
    assert isinstance(provider, IndexedSourceDatasetProvider)
    return provider


def test_each_index_split_becomes_its_source_rows_plus_index_only_columns(tmp_path: Path) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path, {
        "train": {"row": [2, 0], "id": ["c", "a"], "duration_s": [1.5, 2.5]},
        "validation": {"row": [3], "id": ["d"], "duration_s": [0.5]},
    })

    dataset = _provider(index, source, {"train": "train", "validation": "train"}).load_dataset()

    assert set(dataset) == {"train", "validation"}
    assert dataset["train"]["id"] == ["c", "a"]
    assert dataset["train"]["audio"] == [b"2", b"0"]
    assert dataset["train"]["row"] == [2, 0]
    assert dataset["train"]["duration_s"] == [1.5, 2.5]
    assert dataset["validation"]["text"] == ["three"]


def test_an_index_pointing_at_the_wrong_rows_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path, {"train": {"row": [1], "id": ["c"]}})

    with pytest.raises(KeyError, match="different id"):
        _provider(index, source, {"train": "train"}).load_dataset()


def test_an_index_without_a_source_is_used_as_is(tmp_path: Path) -> None:
    index = _index(tmp_path, {"train": {"row": [0], "id": ["a"], "audio": [b"0"]}})

    dataset = _provider(index, None, {"train": "train"}).load_dataset()

    assert dataset["train"].column_names == ["row", "id", "audio"]
