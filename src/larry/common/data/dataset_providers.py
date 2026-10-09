import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence, Mapping
from pathlib import Path
from typing import Any, ClassVar
from typing import Union

import datasets
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset, NamedSplit
from datasets import Split

from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class DatasetProvider[C: DatasetConfig = DatasetConfig](Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[DatasetProvider]]

    _config: C

    def __init__(self, config: C) -> None:
        self._config = config
        self.log = logging.getLogger(type(self).__name__)

    @property
    def config(self) -> C:
        return self._config

    @abstractmethod
    def load_dataset(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        ...


class HuggingFaceDatasetProvider[C: DatasetConfig = DatasetConfig](DatasetProvider[C]):
    @property
    def path(self) -> str | None:
        return self.config.path

    @property
    def name(self) -> str | None:
        return self.config.name

    @property
    def data_dir(self) -> str | None:
        return self.config.data_dir

    @property
    def data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        files = self.config.data_files
        split = self.config.split
        if files is None or isinstance(files, Mapping):
            return files
        if isinstance(split, (str, NamedSplit)):
            return {str(split): files}
        return files

    @property
    def split(self) -> str | Split | list[str] | list[Split] | None:
        return self.config.split

    @property
    def cache_dir(self) -> str | Path | None:
        return self.config.cache_dir

    @property
    def keep_in_memory(self) -> bool | None:
        return self.config.keep_in_memory

    @property
    def streaming(self) -> bool:
        return self.config.streaming

    @property
    def num_proc(self) -> int | None:
        return self.config.num_proc

    @property
    def load_dataset_kwargs(self) -> dict[str, Any]:
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
        kwargs = dict(self.load_dataset_kwargs)
        kwargs.pop("streaming", None)
        self.log.info(f"saving dataset using kwargs={kwargs} from config={self.config}")
        if self.streaming:
            return datasets.load_dataset(**kwargs, streaming=True)
        return datasets.load_dataset(**kwargs, streaming=False)
