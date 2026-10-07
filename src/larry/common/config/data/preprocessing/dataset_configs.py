import dataclasses
from abc import abstractmethod, ABC
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from datasets import Split, Image, Audio

from larry.common.config.data.downloader_configs import DownloaderConfig
from larry.common.config.data.extractor_configs import ExtractorConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class DatasetConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[DatasetConfig]]

    provider_name: str = "HuggingFaceDatasetProvider"
    path: str
    name: str | None = None
    data_dir: str | None = None
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    split: str | Split | list[str] | list[Split] | None = "train"
    cache_dir: str | Path | None = None
    keep_in_memory: bool | None = None
    streaming: bool = False
    num_proc: int | None = None

    provenance_columns: list[str] = dataclasses.field(default_factory=list)
    provenance_ancestry: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class CachedDatasetConfig(DatasetConfig, ABC):
    path: str = "parquet"

    download_extract_cache_dir: str

    batch_size: int = 2000
    cached_parquet_size_mb: int = 500

    @abstractmethod
    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        ...

    @abstractmethod
    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        ...

    download_extract_file_type: str
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    split_column: str | None = None
    split_column_mappings: dict[str, str] | None = None

    select_columns_from_extracted: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class MediaCachedDatasetConfig(CachedDatasetConfig, ABC):
    media_dir: str
    media_type: Audio | Image
    media_column: str
    media_path_column: str = "path"
