import csv
import dataclasses
from typing import Any, ClassVar

from datasets import Audio, Image

from larry.common.config.dataset_config import DatasetConfig, \
    MediaCachedDatasetConfig, ExtractorConfig


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

    AUDIO_FILE: ClassVar[str] = "downloaded.tar.gz"
    ARCHIVE_ROOT: ClassVar[str] = "cv-corpus-27.0-2026-09-11/en"

    provider_name: str = "CommonVoiceMediaCachedDatasetProvider"

    extractor_configs: list[ExtractorConfig] = dataclasses.field(default_factory=lambda: [
        ExtractorConfig(
            input_file_name=CommonVoiceMediaCachedDatasetConfig.AUDIO_FILE,
            archive_root=CommonVoiceMediaCachedDatasetConfig.ARCHIVE_ROOT,
        ),
    ])

    download_extract_file_type: str = "csv"
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=lambda: dict(
        delimiter="\t",
        quoting=csv.QUOTE_NONE
    ))

    media_dir: str = ARCHIVE_ROOT
    media_type: Audio | Image | None = dataclasses.field(default_factory=lambda: Audio(decode=False))
    media_column: str | None = "audio"
