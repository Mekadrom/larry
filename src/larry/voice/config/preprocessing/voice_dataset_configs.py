import csv
import dataclasses
from collections.abc import Sequence
from typing import Any

from datasets import Audio, Image, Split

from larry.common.config.data.downloader_configs import DownloaderConfig
from larry.common.config.data.extractor_configs import SingleFileExtractorConfig, ExtractorConfig
from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig, MediaCachedDatasetConfig, \
    CachedDatasetConfig
from larry.voice.config.preprocessing.voice_downloader_configs import SCOTUSTermManifestDownloaderConfig, \
    SCOTUSDocketManifestDownloaderConfig


@dataclasses.dataclass(kw_only=True)
class VoiceDatasetConfig:
    ...


@dataclasses.dataclass(kw_only=True)
class MythicInfinityLibriHeavyDatasetConfig(DatasetConfig, VoiceDatasetConfig):
    """https://huggingface.co/datasets/mythicinfinity/libriheavy"""

    path: str = "mythicinfinity/libriheavy"
    name: str | None = "large"


@dataclasses.dataclass(kw_only=True)
class CommonVoiceDatasetConfig(MediaCachedDatasetConfig, VoiceDatasetConfig):
    """https://mozilladatacollective.com/datasets/cmu5jplf300nwmh07iqvk9leo"""

    audio_file: str = "downloaded.tar.gz"
    archive_root: str = "cv-corpus-27.0-2026-09-11/en"

    provider_name: str = "CommonVoiceMediaCachedDatasetProvider"

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return []

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return [
            ("SingleFileExtractor", SingleFileExtractorConfig(
                input_file_name=self.audio_file,
                archive_root=self.archive_root,
            )),
        ]

    download_extract_file_type: str = "csv"
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=lambda: dict(
        delimiter="\t",
        quoting=csv.QUOTE_NONE
    ))

    media_dir: str = archive_root
    media_type: Audio | Image = dataclasses.field(default_factory=lambda: Audio(decode=False))
    media_column: str = "audio"


@dataclasses.dataclass(kw_only=True)
class SCOTUSManifestCachedDatasetConfig(CachedDatasetConfig, VoiceDatasetConfig):
    path: str = "parquet"
    base_url: str = "https://www.supremecourt.gov/"
    provider_name: str = "SCOTUSManifestCachedDatasetProvider"

    term_manifest_file_name: str = "term_manifest.tsv"
    docket_manifest_file_name: str = "docket_manifest.tsv"

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return [
            ("SCOTUSTermManifestDownloader", SCOTUSTermManifestDownloaderConfig(
                base_url=self.base_url,
                term_manifest_file_name=self.term_manifest_file_name,
                docket_manifest_file_name=self.docket_manifest_file_name,
                range_desc=self.terms,
                busy_wait=1.0,
            )),
            ("SCOTUSDocketManifestDownloader", SCOTUSDocketManifestDownloaderConfig(
                base_url=f"{self.base_url}",
                term_manifest_file_name=self.term_manifest_file_name,
                docket_manifest_file_name=self.docket_manifest_file_name,
            )),
        ]

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return []

    download_extract_file_type: str = "csv"
    download_extract_transform_kwargs: dict[str, Any] = dataclasses.field(default_factory=lambda: dict(
        delimiter="\t",
        quoting=csv.QUOTE_NONE
    ))

    sleep: float = 1.0

    terms: str = "2010-2025"


@dataclasses.dataclass(kw_only=True)
class LarrySCOTUSManifestDatasetConfig(DatasetConfig, VoiceDatasetConfig):
    """https://huggingface.co/datasets/thelarryproject/scotus-manifest"""

    path: str = "thelarryproject/scotus-manifest"
