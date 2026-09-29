import dataclasses
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Split, Image, Audio

log = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class DatasetConfig:
    """Base class for all config DTOs having to do with a dataset; no functionality."""

    provider_name: str | None = "HuggingFaceDatasetProvider"
    path: str | None = None
    name: str | None = None
    data_dir: str | None = None
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    split: str | Split | list[str] | list[Split] | None = "train"
    cache_dir: str | Path | None = None
    keep_in_memory: bool | None = None
    streaming: bool = False
    num_proc: int | None = None

    provenance_columns: list[str] = dataclasses.field(default_factory=list)


def _strip_archive_suffix(file_name: str) -> str:
    for suffix in (".tar.gz", ".tgz", ".tar.xz", ".tar", ".zip"):
        if file_name.endswith(suffix):
            return file_name.removesuffix(suffix)
    log.info(
        f"Did not remove suffix from file_name={file_name}, considering adding this suffix: {file_name.split('.')[-1]}")
    return file_name


class DownloaderConfig:
    def __init__(
            self,
            download_url: str,
            output_file_name: str,
            download_path: str | None = None,
            download_timeout: int = 10,
    ) -> None:
        self.download_url = download_url
        self.output_file_name = output_file_name
        self.download_path = download_path or output_file_name
        self.download_timeout = download_timeout


class ExtractorConfig:
    def __init__(
            self,
            input_file_name: str,
            archive_root: str | None = None,
    ) -> None:
        self.input_file_name = input_file_name
        self.archive_root = archive_root or _strip_archive_suffix(input_file_name)


@dataclasses.dataclass(kw_only=True)
class CachedDatasetConfig(DatasetConfig):
    provider_name: str | None = None
    download_extract_cache_dir: str

    batch_size: int = 2000
    cached_parquet_size_mb: int = 500

    downloader_configs: list[DownloaderConfig] = dataclasses.field(default_factory=list)
    extractor_configs: list[ExtractorConfig] = dataclasses.field(default_factory=list)

    download_extract_file_type: str
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    split_column: str | None = None
    split_column_mappings: dict[str, str] | None = None

    select_columns_from_extracted: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class MediaCachedDatasetConfig(CachedDatasetConfig):
    media_dir: str
    media_type: Audio | Image | None
    media_column: str | None
    media_path_column: str = "path"
