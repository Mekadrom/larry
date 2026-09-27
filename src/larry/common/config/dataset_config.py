import dataclasses
import logging
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from datasets import Split, Image, Audio

from larry.utils.types import StringRegistry

log = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class DatasetConfig:
    """Base class for all config DTOs having to do with a dataset; no functionality."""

    provider_name: str | None = "HuggingFaceDatasetProvider"
    path: str | None = None
    name: str | None = None
    data_dir: str | None = None
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    split: str | Split | list[str] | list[Split] | None = None
    cache_dir: str | Path | None = None
    keep_in_memory: bool | None = None
    streaming: bool = False
    num_proc: int | None = 8

    provenance_columns: list[str] = dataclasses.field(default_factory=list)


class NotSupported(Enum):
    NOT_SUPPORTED = "NOT_SUPPORTED"


NOT_SUPPORTED = NotSupported.NOT_SUPPORTED


def default_if_supported[T](value: T | NotSupported | None, default_value: T | None) -> T | None:
    match value:
        case NotSupported.NOT_SUPPORTED:
            return None
        case None:
            return default_value
    return value


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
            download_path: str,
            override_download_base_url: str,
            output_file_name: str,
    ) -> None:
        self.download_path = download_path or output_file_name
        self.override_download_base_url = override_download_base_url

        if self.override_download_base_url is not None and not self.override_download_base_url.endswith("/"):
            self.override_download_base_url = self.override_download_base_url + "/"

        self.output_file_name = output_file_name


class ExtractorConfig:
    def __init__(
            self,
            intput_file_name: str,
            archive_root: str | NotSupported | None = None,
    ) -> None:
        self.intput_file_name = intput_file_name
        self.archive_root = archive_root or _strip_archive_suffix(intput_file_name)


class DownloadExtractConfig:
    def __init__(
            self,
            file_name: str,
            download_path: str | NotSupported | None = None,
            archive_root: str | NotSupported | None = None,
            override_download_base_url: str | NotSupported | None = None,
    ) -> None:
        self.file_name = file_name

        self.download_path = default_if_supported(download_path, file_name)
        self.archive_root = default_if_supported(archive_root, _strip_archive_suffix(file_name))
        self.override_download_base_url = default_if_supported(override_download_base_url, None)

        if self.override_download_base_url is not None and not self.override_download_base_url.endswith("/"):
            self.override_download_base_url = self.override_download_base_url + "/"

    @classmethod
    def create_registry(cls, configs: list[DownloadExtractConfig]) -> StringRegistry[DownloadExtractConfig]:
        return {c.file_name: c for c in configs}


@dataclasses.dataclass(kw_only=True)
class CachedDatasetConfig(DatasetConfig):
    provider_name: str | None = None
    download_extract_cache_dir: str

    batch_size: int = 2000
    download_timeout: int = 10
    cached_parquet_size_mb: int = 500
    default_download_url: str | None = None

    download_extract_configs: StringRegistry[DownloadExtractConfig] = dataclasses.field(default_factory=dict)
    download_extract_file_type: str
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    select_columns_from_extracted: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class MediaCachedDatasetConfig(CachedDatasetConfig):
    media_type: Audio | Image | None
    media_file_name: str | None = None
    media_column: str | None
    media_path_column: str = "path"
