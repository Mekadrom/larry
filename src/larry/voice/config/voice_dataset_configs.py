import csv
import dataclasses
from typing import Any, ClassVar

from datasets import Audio, Image

from larry.common.config.dataset_config import DatasetConfig, DownloadExtractConfig, NotSupported, \
    MediaCachedDatasetConfig
from larry.utils.types import StringRegistry


@dataclasses.dataclass
class VoiceDatasetConfig(DatasetConfig):
    """Super class for all DTOs related to voice dataset preprocessing."""


@dataclasses.dataclass(kw_only=True)
class MythicInfinityLibriHeavyDatasetConfig(VoiceDatasetConfig):
    """https://huggingface.co/datasets/mythicinfinity/libriheavy"""

    path: str | None = "mythicinfinity/libriheavy"
    name: str | None = "large"
    split: str | None = "train"


@dataclasses.dataclass(kw_only=True)
class CommonVoiceMediaCachedDatasetConfig(VoiceDatasetConfig, MediaCachedDatasetConfig):
    """https://mozilladatacollective.com/datasets/cmu5jplf300nwmh07iqvk9leo"""

    AUDIO_FILE: ClassVar[str] = "download.tar.gz"

    provider_name: str = "CommonVoiceMediaCachedDatasetProvider"

    download_extract_configs: StringRegistry[DownloadExtractConfig] = dataclasses.field(
        default_factory=lambda: DownloadExtractConfig.create_registry([
            DownloadExtractConfig(
                CommonVoiceMediaCachedDatasetConfig.AUDIO_FILE,
                download_path=NotSupported.NOT_SUPPORTED,
                archive_root="cv-corpus-27.0-2026-09-11/en"
            )
        ])
    )
    download_extract_file_type: str = "csv"
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "delimiter": "\t",
        "quoting": csv.QUOTE_NONE,
    })

    media_type: Audio | Image | None = dataclasses.field(default_factory=Audio)
    media_file_name: str | None = AUDIO_FILE
    media_column: str | None = "audio"
