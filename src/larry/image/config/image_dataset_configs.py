import dataclasses
from typing import ClassVar

from datasets import Image, Audio

from larry.common.config.dataset_config import DatasetConfig, DownloadExtractConfig, NotSupported, \
    MediaCachedDatasetConfig
from larry.utils.types import StringRegistry


@dataclasses.dataclass
class ImageDatasetConfig(DatasetConfig):
    """Super class for all DTOs related to image dataset preprocessing."""


@dataclasses.dataclass(kw_only=True)
class GoogleDOCCIMediaCachedDatasetConfig(ImageDatasetConfig, MediaCachedDatasetConfig):
    """https://huggingface.co/datasets/google/docci"""

    DESCRIPTIONS_FILE: ClassVar[str] = "docci_descriptions.jsonlines"
    IMAGES_FILE: ClassVar[str] = "docci_images.tar.gz"

    provider_name: str = "GoogleDOCCIMediaCachedDatasetProvider"
    default_download_url: str | None = "https://storage.googleapis.com/docci/data"

    download_extract_configs: StringRegistry[DownloadExtractConfig] = dataclasses.field(
        default_factory=lambda: DownloadExtractConfig.create_registry([
            DownloadExtractConfig(
                GoogleDOCCIMediaCachedDatasetConfig.DESCRIPTIONS_FILE,
                archive_root=NotSupported.NOT_SUPPORTED
            ),
            DownloadExtractConfig(
                GoogleDOCCIMediaCachedDatasetConfig.IMAGES_FILE,
                archive_root="images"
            )
        ])
    )
    download_extract_file_type: str = "json"

    media_type: Audio | Image | None = dataclasses.field(default_factory=Image)
    media_file_name: str | None = IMAGES_FILE
    media_column: str | None = "image"
