import dataclasses
from typing import ClassVar

from datasets import Image, Audio

from larry.common.config.dataset_config import DatasetConfig, MediaCachedDatasetConfig, DownloaderConfig, \
    ExtractorConfig


@dataclasses.dataclass
class ImageDatasetConfig(DatasetConfig):
    """Super class for all DTOs related to image dataset preprocessing."""


@dataclasses.dataclass(kw_only=True)
class GoogleDOCCIMediaCachedDatasetConfig(ImageDatasetConfig, MediaCachedDatasetConfig):
    """https://huggingface.co/datasets/google/docci"""

    DOWNLOAD_URL: ClassVar[str] = "https://storage.googleapis.com/docci/data"
    DESCRIPTIONS_FILE: ClassVar[str] = "docci_descriptions.jsonlines"
    IMAGES_FILE: ClassVar[str] = "docci_images.tar.gz"
    ARCHIVE_ROOT: ClassVar[str] = "images"

    provider_name: str = "GoogleDOCCIMediaCachedDatasetProvider"

    downloader_configs: list[DownloaderConfig] = dataclasses.field(default_factory=lambda: [
        DownloaderConfig(
            download_url=GoogleDOCCIMediaCachedDatasetConfig.DOWNLOAD_URL,
            output_file_name=GoogleDOCCIMediaCachedDatasetConfig.DESCRIPTIONS_FILE,
        ),
        DownloaderConfig(
            download_url=GoogleDOCCIMediaCachedDatasetConfig.DOWNLOAD_URL,
            output_file_name=GoogleDOCCIMediaCachedDatasetConfig.IMAGES_FILE,
        ),
    ])
    extractor_configs: list[ExtractorConfig] = dataclasses.field(default_factory=lambda: [
        ExtractorConfig(
            input_file_name=GoogleDOCCIMediaCachedDatasetConfig.IMAGES_FILE,
            archive_root=GoogleDOCCIMediaCachedDatasetConfig.ARCHIVE_ROOT
        )
    ])

    download_extract_file_type: str = "json"

    split_column: str | None = "split"

    media_dir: str = ARCHIVE_ROOT
    media_type: Audio | Image | None = dataclasses.field(default_factory=lambda: Image(decode=False))
    media_column: str | None = "image"
    media_path_column: str = "image_file"
