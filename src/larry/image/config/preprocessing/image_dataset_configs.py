import dataclasses
from collections.abc import Sequence

from datasets import Image, Audio

from larry.common.config.data.downloader_configs import MultiFileDownloaderConfig, SingleFileDownloaderConfig, \
    DownloaderConfig
from larry.common.config.data.extractor_configs import SingleFileExtractorConfig, ExtractorConfig
from larry.common.config.data.preprocessing.dataset_configs import MediaCachedDatasetConfig, DatasetConfig


@dataclasses.dataclass
class ImageDatasetConfig(DatasetConfig):
    ...


@dataclasses.dataclass(kw_only=True)
class GoogleDOCCIMediaCachedDatasetConfig(ImageDatasetConfig, MediaCachedDatasetConfig):
    """https://huggingface.co/datasets/google/docci"""

    download_url: str = "https://storage.googleapis.com/docci/data"
    descriptions_file: str = "docci_descriptions.jsonlines"
    images_file: str = "docci_images.tar.gz"
    archive_root: str = "images"

    provider_name: str = "GoogleDOCCIMediaCachedDatasetProvider"

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
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
        return [
            ("SingleFileExtractor", SingleFileExtractorConfig(
                input_file_name=self.images_file,
                archive_root=self.archive_root,
            )),
        ]

    download_extract_file_type: str = "json"

    split_column: str | None = "split"

    media_dir: str = archive_root
    media_type: Audio | Image = dataclasses.field(default_factory=lambda: Image(decode=False))
    media_column: str = "image"
    media_path_column: str = "image_file"
