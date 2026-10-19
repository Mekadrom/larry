import bisect
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

from larry.common.config.data.data_configs import DatasetSourceConfig


def index_split_files(index_path: str, split: str) -> list[str]:
    """Parses combined split definitions in order to provide all files from these splits.

    Works on split strings like "train.clean+train.other".
    """
    files: list[str] = []
    for name in split.split("+"):
        found = sorted(glob.glob(str(Path(index_path).expanduser() / name / "*.parquet")))
        if not found:
            raise FileNotFoundError(f"no parquet files for split {name!r} under {index_path}")
        files.extend(found)
    return files


def load_index(config: DatasetSourceConfig) -> HFDataset:
    """Helper for loading an index dataset from either local parquets or huggingface."""
    if Path(config.index_path).expanduser().is_dir():
        return load_dataset(
            "parquet",
            data_files=index_split_files(config.index_path, config.split),
            split="train"
        )

    return load_dataset(config.index_path, split=config.split)


def load_source(config: DatasetSourceConfig) -> HFDataset | None:
    if config.source_path is None:
        return None

    return load_dataset(
        config.source_path,
        name=config.source_name,
        split=config.source_split,
        revision=config.source_revision,
        cache_dir=config.source_cache_dir,
    )


def load_source_for_columns(config: DatasetSourceConfig, index: HFDataset, columns: list[str]) -> HFDataset | None:
    if set(columns) <= set(index.column_names):
        return None

    return load_source(config)


def arrow_column(dataset: HFDataset, name: str) -> pa.ChunkedArray:
    return cast(pa.Table, dataset.select_columns([name]).with_format("arrow")[:]).column(name)


def indexed_column(index: HFDataset, source: HFDataset | None, name: str) -> pa.ChunkedArray:
    if name in index.column_names:
        return arrow_column(index, name)

    if source is None:
        raise KeyError(f"column {name!r} is not in the index and there is no source to read it from")

    return arrow_column(source, name).take(arrow_column(index, "row"))


class IndexedDataset[S: DatasetSourceConfig = DatasetSourceConfig](Dataset[dict[str, Any]], ABC):
    """Specific dataset type for datasets that are represented by a large materialized form (source) and an index."""

    def __init__(
            self,
            configs: list[S],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
    ) -> None:
        """Initializes index offsets."""
        if not sources:
            raise ValueError("sources are required")

        self.configs = configs
        self.indexes = indexes
        self.sources = sources

        self.offsets = [0]
        for index in indexes:
            self.offsets.append(self.offsets[-1] + len(index))

    def __len__(self) -> int:
        """Returns the final offset index as the length."""
        return self.offsets[-1]

    def __getitem__(self, k: int) -> dict[str, Any]:
        """Returns a single item as a dictionary from the requested index, determining which source it falls in."""
        source_idx = bisect.bisect_right(self.offsets, k) - 1
        config = self.configs[source_idx]
        entry = self.indexes[source_idx][k - self.offsets[source_idx]]

        source = self.sources[source_idx]
        if source is not None:
            row = source[entry["row"]]
            if row["id"] != entry["id"]:
                raise KeyError(f"{config.name} row {entry['row']} is {row['id']}, index expects {entry['id']}")
            entry = {**row, **entry}

        return self.format_example_dict(source_idx, entry, self.get_data(source_idx, entry))

    def column(self, source_idx: int, name: str) -> pa.ChunkedArray:
        return indexed_column(self.indexes[source_idx], self.sources[source_idx], name)

    @abstractmethod
    def get_data(self, source_idx: int, entry: dict[Any, Any]) -> Any:
        ...

    @abstractmethod
    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        ...
