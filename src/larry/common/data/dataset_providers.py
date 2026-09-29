import abc
import dataclasses
import logging
from abc import abstractmethod
from typing import Sequence, Mapping, Any
from typing import Union

import datasets
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets import Split


@dataclasses.dataclass(kw_only=True)
class DatasetProvider[C](abc.ABC):
    """Wrapper interface that to provide ``DatasetDict | Dataset | IterableDatasetDict | IterableDataset``."""
    config: C

    def __init__(self, config: C) -> None:
        self.config = config
        self.log = logging.getLogger(type(self).__name__)

    @abstractmethod
    def load_dataset(self) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """Loads a dataset and returns it in the huggingface format."""
        ...


class HuggingFaceDatasetProvider[C](DatasetProvider[C]):
    """Default implementation of ``DatasetProvider`` which loads using huggingface's datasets library."""
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
        return {self.split: self.config.data_files}

    @property
    def split(self) -> str | Split | list[str] | list[Split] | None:
        return self.config.split

    @property
    def cache_dir(self) -> str | None:
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
        """Delegates to huggingface's ``datasets.load_dataset()``, passing config values in verbatim.

        Has to switch on ``self.config.streaming`` because the overloads of ``datasets.load_dataset()`` mess with the
        type hinting when passing a ``bool`` typed field instead of a literal ``True`` or ``False``.
        """
        kwargs = dict(self.load_dataset_kwargs)
        kwargs.pop("streaming", None)
        self.log.info(f"saving dataset using kwargs={kwargs} from config={self.config}")
        if self.streaming:
            return datasets.load_dataset(**kwargs, streaming=True)
        return datasets.load_dataset(**kwargs, streaming=False)
