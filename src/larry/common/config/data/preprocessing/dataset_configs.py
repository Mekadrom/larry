import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Audio, Image

from larry.common.config.config import TaggedConfig
from larry.common.config.data.downloader_configs import DownloaderConfig
from larry.common.config.data.extractor_configs import ExtractorConfig


@dataclasses.dataclass(kw_only=True)
class DatasetConfig(TaggedConfig):
    """Superclass for configs for all dataset providers for preprocessing pipelines only."""

    path: str
    name: str | None = None
    data_dir: str | None = None
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    split: str | list[str] | None = "train"
    cache_dir: str | Path | None = None
    keep_in_memory: bool | None = None
    streaming: bool = False
    num_proc: int | None = None


@dataclasses.dataclass(kw_only=True)
class DefaultDatasetConfig(DatasetConfig):
    """Default impl for dataset providers that use the simple huggingface route for providing a dataset."""

    ...


@dataclasses.dataclass(kw_only=True)
class CachedDatasetConfig(DatasetConfig, ABC):
    """Superclass for configs for any dataset provider that requires downloading or extracting prior to preprocessing.

    Used to replace any datasets that originally required arbitrary python code execution, essentially.
    """

    path: str = "parquet"
    split: str | list[str] | None = None

    download_extract_cache_dir: str

    batch_size: int = 2000
    cached_parquet_size_mb: int = 500

    @abstractmethod
    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        """Configure in subclasses to return a ``Sequence`` of tuples of (``str``, ``DownloaderConfig``).

        Each downloader is to be instantiated using the name from the string in the first position in each tuple using
        the config in the second position of the same tuple, and the downloaders then run sequentially in insertion
        order.
        """
        ...

    @abstractmethod
    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        """Configure in subclasses to return a ``Sequence`` of tuples of (``str``, ``ExtractorConfig``).

        Each extractor is to be instantiated using the name from the string in the first position in each tuple using
        the config in the second position of the same tuple, and the extractors then run sequentially in insertion
        order.
        """
        ...

    download_extract_file_type: str
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    split_column: str | None = None
    split_column_mappings: dict[str, str] | None = None

    select_columns_from_extracted: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class MediaCachedDatasetConfig(CachedDatasetConfig, ABC):
    """Super class for all implementations of ``CachedDatasetConfig`` which require media processing.

    E.g. images or audio.
    """

    media_dir: str
    media_type: Audio | Image
    media_column: str
    media_path_column: str = "path"
