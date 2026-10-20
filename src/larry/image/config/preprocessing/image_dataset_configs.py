import dataclasses
from collections.abc import Sequence

from datasets import Image

from larry.common.config.data.downloader_configs import (
    DownloaderConfig,
    MultiFileDownloaderConfig,
    SingleFileDownloaderConfig,
)
from larry.common.config.data.extractor_configs import ExtractorConfig, SingleFileExtractorConfig
from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig, MediaCachedDatasetConfig


@dataclasses.dataclass(kw_only=True)
class ImageDatasetConfig(DatasetConfig):
    """Superclass config for all image dataset configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class GoogleDOCCIMediaCachedDatasetConfig(MediaCachedDatasetConfig, ImageDatasetConfig):
    """Loads the google/docci dataset from its hosted location, not from huggingface.

    https://huggingface.co/datasets/google/docci - the documentation this references (including python script).
    """

    download_url: str = "https://storage.googleapis.com/docci/data"
    descriptions_file: str = "docci_descriptions.jsonlines"
    images_file: str = "docci_images.tar.gz"
    archive_root: str = "images"

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        """Configures downloaders for both files from google/docci

        Configures downloaders for the jsonlines descriptions file and the docci_images.tar.gz archive.
        """
        return [
            ("MultiFileDownloader", MultiFileDownloaderConfig(
                configs=[
                    ("SingleFileDownloaderConfig", SingleFileDownloaderConfig(
                        download_base_url=self.download_url,
                        output_file_name=self.descriptions_file,
                    )),
                    ("SingleFileDownloaderConfig", SingleFileDownloaderConfig(
                        download_base_url=self.download_url,
                        output_file_name=self.images_file,
                    )),
                ],
            )),
        ]

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        """Configures extraction for the images archive."""
        return [
            ("SingleFileExtractor", SingleFileExtractorConfig(
                input_file_name=self.images_file,
                archive_root=self.archive_root,
            )),
        ]

    download_extract_file_type: str = "json"

    split_column: str | None = "split"

    media_dir: str = ""
    media_type: Image = dataclasses.field(default_factory=lambda: Image(decode=False))  # pyright: ignore[reportIncompatibleVariableOverride]
    media_column: str = "image"
    media_path_column: str = "image_file"

    def __post_init__(self) -> None:
        """Defaults ``media_dir`` to ``archive_root`` after any overrides of either are applied."""
        self.media_dir = self.media_dir or self.archive_root
