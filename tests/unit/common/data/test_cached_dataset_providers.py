import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from datasets import DatasetDict

from larry.common.config.data.downloader_configs import DownloaderConfig
from larry.common.config.data.extractor_configs import ExtractorConfig
from larry.common.config.data.preprocessing.dataset_configs import CachedDatasetConfig
from larry.common.data.cached_dataset_providers import CachedDatasetProvider


@dataclasses.dataclass(kw_only=True)
class ToyCachedDatasetConfig(CachedDatasetConfig):
    download_extract_file_type: str = "csv"
    input_files: dict[str, str] = dataclasses.field(default_factory=dict)

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return []

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return []


class ToyCachedDatasetProvider(CachedDatasetProvider[ToyCachedDatasetConfig]):
    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return self.config.input_files

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return batch

    def cast_cols(self, dataset: Any) -> Any:
        return dataset


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> str:
    path.write_text("text,split\n" + "".join(f"{text},{split}\n" for text, split in rows))
    return str(path)


def _provider(tmp_path: Path, input_files: dict[str, str], **overrides: Any) -> ToyCachedDatasetProvider:
    return ToyCachedDatasetProvider(ToyCachedDatasetConfig(
        download_extract_cache_dir=str(tmp_path / "download_extract"),
        cache_dir=str(tmp_path / "cache"),
        input_files=input_files,
        **overrides,
    ))


def _three_files(tmp_path: Path) -> dict[str, str]:
    return {
        "train": _write_csv(tmp_path / "train.csv", [("a", "x"), ("b", "x"), ("c", "x")]),
        "dev": _write_csv(tmp_path / "dev.csv", [("d", "x")]),
        "test": _write_csv(tmp_path / "test.csv", [("e", "x"), ("f", "x")]),
    }


def test_every_input_file_becomes_its_own_split(tmp_path: Path) -> None:
    dataset = _provider(tmp_path, _three_files(tmp_path)).load_dataset()

    assert isinstance(dataset, DatasetDict)
    assert {name: sorted(split["text"]) for name, split in dataset.items()} == {
        "train": ["a", "b", "c"],
        "dev": ["d"],
        "test": ["e", "f"],
    }


def test_split_column_still_splits_a_single_input_file(tmp_path: Path) -> None:
    files = {"train": _write_csv(tmp_path / "all.csv", [("a", "train"), ("b", "val"), ("c", "train")])}

    dataset = _provider(tmp_path, files, split_column="split", split_column_mappings={"val": "validation"}).load_dataset()

    assert isinstance(dataset, DatasetDict)
    assert {name: sorted(split["text"]) for name, split in dataset.items()} == {
        "train": ["a", "c"],
        "validation": ["b"],
    }


def test_split_column_with_several_input_files_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _provider(tmp_path, _three_files(tmp_path), split_column="split").load_dataset()


def test_missing_input_file_is_reported_with_its_split(tmp_path: Path) -> None:
    files = {**_three_files(tmp_path), "dev": str(tmp_path / "validation.csv")}

    with pytest.raises(FileNotFoundError, match="dev"):
        _provider(tmp_path, files).load_dataset()


def test_completed_split_is_not_read_again(tmp_path: Path) -> None:
    files = _three_files(tmp_path)
    _provider(tmp_path, files).load_dataset()
    Path(files["train"]).unlink()

    dataset = _provider(tmp_path, files).load_dataset()

    assert isinstance(dataset, DatasetDict)
    assert sorted(dataset["train"]["text"]) == ["a", "b", "c"]


def test_split_without_a_completion_marker_is_not_loaded(tmp_path: Path) -> None:
    provider = _provider(tmp_path, _three_files(tmp_path))
    provider.load_dataset()
    (provider.parquet_dir / "test.parquets.complete").unlink()

    assert sorted(provider.data_files) == ["dev", "train"]
