import dataclasses
from collections.abc import Sequence

from datasets import Image

from larry.common.config.data.downloader_configs import (
    DownloaderConfig,
    MultiFileDownloaderConfig,
    SingleFileDownloaderConfig,
)
from larry.common.config.data.extractor_configs import ExtractorConfig, SingleFileExtractorConfig
from larry.common.config.data.preprocessing.dataset_configs import (
    DatasetConfig,
    FilteredMetadataCachedDatasetConfig,
    MediaCachedDatasetConfig,
)


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
                    ("SingleFileDownloader", SingleFileDownloaderConfig(
                        download_base_url=self.download_url,
                        output_file_name=self.descriptions_file,
                    )),
                    ("SingleFileDownloader", SingleFileDownloaderConfig(
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


@dataclasses.dataclass(kw_only=True)
class WikimediaCommonsCachedDatasetConfig(FilteredMetadataCachedDatasetConfig, ImageDatasetConfig):
    dumps_url: str = "https://dumps.wikimedia.org"
    dump_date: str = "20261019"
    upload_url: str = "https://upload.wikimedia.org/wikipedia/commons"
    site_url: str = "https://commons.wikimedia.org/wiki"
    metadata_file: str = "commons-mediainfo-filtered.parquet"

    licenses: list[str] = dataclasses.field(default_factory=lambda: ["Q6938433"])
    copyright_statuses: list[str] = dataclasses.field(default_factory=lambda: ["Q19652"])
    mime_types: list[str] = dataclasses.field(default_factory=lambda: ["image/jpeg", "image/png", "image/webp"])
    require_dimensions: bool = True
    thumbnail_width: int = 960
    deduplicate_ids: bool = False

    @property
    def mediainfo_file(self) -> str:
        return f"commons-{self.dump_date}-mediainfo.json.gz"

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return [
            ("SingleFileDownloader", SingleFileDownloaderConfig(
                download_base_url=self.dumps_url,
                download_path=f"other/wikibase/commonswiki/{self.dump_date}/{self.mediainfo_file}",
                output_file_name=self.mediainfo_file,
                download_timeout=60,
                max_retries=5,
                busy_wait=30.0,
            )),
        ]

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return []


@dataclasses.dataclass(kw_only=True)
class SmithsonianOpenAccessCachedDatasetConfig(FilteredMetadataCachedDatasetConfig, ImageDatasetConfig):
    bucket_url: str = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com"
    metadata_prefix: str = "metadata/edan"
    ids_url: str = "https://ids.si.edu/ids/deliveryService"
    metadata_file: str = "smithsonian-open-access-filtered.parquet"

    units: list[str] | None = None
    excluded_units: list[str] = dataclasses.field(default_factory=list)
    media_types: list[str] = dataclasses.field(default_factory=lambda: ["Images"])
    usages: list[str] = dataclasses.field(default_factory=lambda: ["CC0"])
    description_labels: list[str] = dataclasses.field(default_factory=lambda: [
        "Description (Brief)", "Description", "Summary", "Gallery Label",
    ])
    max_size: int = 960
    keep_raw_metadata: bool = False

    @property
    def index_file(self) -> str:
        return f"{self.metadata_prefix}/index.txt"

    def file_downloader_config(self, path: str) -> SingleFileDownloaderConfig:
        return SingleFileDownloaderConfig(
            download_base_url=self.bucket_url,
            download_path=path,
            output_file_name=path,
            download_timeout=60,
            max_retries=5,
            busy_wait=5.0,
        )

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return [("SingleFileDownloader", self.file_downloader_config(self.index_file))]

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return []
