import bisect
import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
        if source is None:
            data = self.get_data(source_idx, entry)
        else:
            row = source[entry["row"]]
            if row["id"] != entry["id"]:
                raise KeyError(f"{config.name} row {entry['row']} is {row['id']}, index expects {entry['id']}")
            data = self.get_data(source_idx, row)

        return self.format_example_dict(source_idx, entry, data)

    @abstractmethod
    def get_data(self, source_idx: int, entry: dict[Any, Any]) -> Any:
        ...

    @abstractmethod
    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        ...
