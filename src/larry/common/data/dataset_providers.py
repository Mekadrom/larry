import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, NamedSplit, Split

from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class DatasetProvider[C: DatasetConfig = DatasetConfig](Registrable, ABC, root=True):
    """Superclass for all dataset providers for training or preprocessing pipelines."""

    REGISTRY: ClassVar[TypeRegistry[DatasetProvider]]

    _config: C

    def __init__(self, config: C) -> None:
        """Just sets up a logger."""
        self.log = logging.getLogger(type(self).__name__)
        self._config = config

    @property
    def config(self) -> C:
        """Immutable type-hint helper for obtaining the config instance."""
        return self._config

    @abstractmethod
    def load_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Primary entrypoint for consumers of the dataset provider.

        The assumption for consumers of this method's outputs is that all work needed to be done prior to preprocessing
        has been done, and that this dataset is backed by parquets or arrow files on disk prior to loading in memory.
        """
        ...


class HuggingFaceDatasetProvider[C: DatasetConfig = DatasetConfig](DatasetProvider[C]):
    """Default implementation of ``DatasetProvider`` that loads the configured dataset from huggingface."""

    @property
    def path(self) -> str | None:
        """Allows overriding of the path getter."""
        return self.config.path

    @property
    def name(self) -> str | None:
        """Allows overriding of the name getter."""
        return self.config.name

    @property
    def data_dir(self) -> str | None:
        """Allows overriding of the data_dir getter."""
        return self.config.data_dir

    @property
    def data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        """Allows overriding of the data_files getter. Also has some helper logic to massage the returned mapping."""
        files = self.config.data_files
        split = self.config.split
        if files is None or isinstance(files, Mapping):
            return files
        if isinstance(split, (str, NamedSplit)):
            return {str(split): files}
        return files

    @property
    def split(self) -> str | Split | list[str] | list[Split] | None:
        """Allows overriding of the split getter."""
        return self.config.split

    @property
    def cache_dir(self) -> str | Path | None:
        """Allows overriding of the cache_dir getter."""
        return self.config.cache_dir

    @property
    def keep_in_memory(self) -> bool | None:
        """Allows overriding of the keep_in_memory getter."""
        return self.config.keep_in_memory

    @property
    def streaming(self) -> bool:
        """Allows overriding of the streaming getter."""
        return self.config.streaming

    @property
    def num_proc(self) -> int | None:
        """Allows overriding of the num_proc getter."""
        return self.config.num_proc

    @property
    def load_dataset_kwargs(self) -> dict[str, Any]:
        """Allows overriding of the entire kwargs passed to load_dataset."""
        return dict(
            path=self.path,
            name=self.name,
            data_dir=self.data_dir,
            data_files=self.data_files,
            split=self.split,
            cache_dir=self.cache_dir,
            keep_in_memory=self.keep_in_memory,
            num_proc=self.num_proc,
        )

    def load_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Switching on ``streaming`` due to type-hinting quirks, loads the dataset by delegating to HF datasets."""
        kwargs = dict(self.load_dataset_kwargs)
        kwargs.pop("streaming", None)

        self.log.info(f"Saving/using dataset using kwargs={kwargs} from config={self.config}")
        if self.streaming:
            return datasets.load_dataset(**kwargs, streaming=True)

        return datasets.load_dataset(**kwargs, streaming=False)
