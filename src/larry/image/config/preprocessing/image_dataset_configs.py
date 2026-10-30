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
    PagedApiCachedDatasetConfig,
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
        "Description (Brief)", "Description", "Long Description", "Summary", "Gallery Label", "Luce Center Label",
        "Exhibition Label", "Label", "Caption", "Physical Description",
    ])
    unit_filters: dict[str, str] = dataclasses.field(default_factory=dict)
    generation_excluded_units: dict[str, str] = dataclasses.field(default_factory=lambda: {
        unit: "specimen" for unit in (
            "nmnhanthro", "nmnhbirds", "nmnhbotany", "nmnheducation", "nmnhento", "nmnhfishes", "nmnhherps",
            "nmnhinv", "nmnhmammals", "nmnhminsci", "nmnhpaleo",
        )
    })
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


@dataclasses.dataclass(kw_only=True)
class ArtInstituteOfChicagoCachedDatasetConfig(FilteredMetadataCachedDatasetConfig, ImageDatasetConfig):
    dump_url: str = "https://artic-api-data.s3.amazonaws.com"
    dump_file: str = "artic-api-data.tar.bz2"
    artworks_prefix: str = "artic-api-data/json/artworks/"
    iiif_url: str = "https://www.artic.edu/iiif/2"
    site_url: str = "https://www.artic.edu/artworks"
    metadata_file: str = "artic-artworks-filtered.parquet"

    image_width: int = 843
    include_alt_images: bool = False

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return [
            ("SingleFileDownloader", SingleFileDownloaderConfig(
                download_base_url=self.dump_url,
                output_file_name=self.dump_file,
                download_timeout=60,
                max_retries=5,
                busy_wait=30.0,
            )),
        ]

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return []


@dataclasses.dataclass(kw_only=True)
class TePapaCachedDatasetConfig(PagedApiCachedDatasetConfig, ImageDatasetConfig):
    api_url: str = "https://data.tepapa.govt.nz/collection"
    api_key_env_var: str | None = "TEPAPA_API_KEY"
    api_key_file: str | None = "~/.config/larry/tepapa_api_key"
    site_url: str = "https://collections.tepapa.govt.nz/object"
    metadata_file: str = "tepapa-objects-filtered.parquet"

    page_size: int = 50
    scroll_minutes: int = 5
    min_request_interval: float = 0.25
    fields: list[str] = dataclasses.field(default_factory=lambda: [
        "id", "type", "collection", "title", "description", "caption", "creditLine", "isMadeOfSummary",
        "production", "depicts", "refersTo", "isTypeOf", "hasRepresentation",
    ])

    licenses: list[str] = dataclasses.field(default_factory=lambda: [
        "No Known Copyright Restrictions", "CC BY 4.0", "CC 0",
    ])
    excluded_collections: list[str] = dataclasses.field(default_factory=list)
    generation_excluded_collections: list[str] = dataclasses.field(
        default_factory=lambda: ["TaongaMāori", "PacificCultures"]
    )
    generation_excluded_terms: dict[str, list[str]] = dataclasses.field(default_factory=lambda: {
        "maori_subject": [
            "Māori", "Maori", "iwi", "hapū", "hapu", "marae", "wharenui", "wharekai", "kāinga", "kainga", "pā",
            "rangatira", "ariki", "moko", "tā moko", "haka", "waka", "tūpuna", "tupuna", "tīpuna", "tipuna",
            "taonga", "tangata whenua", "whakairo", "korowai", "hei tiki", "tiki", "taiaha", "pounamu", "tukutuku",
            "kōwhaiwhai", "kowhaiwhai", "karakia",
        ],
        "specimen": [
            "specimen", "specimens", "fossil", "fossils", "herbarium", "skeleton", "skeletons", "skull", "skulls",
            "taxidermy", "x-ray", "x-rays", "radiograph", "radiographs",
        ],
    })
    max_images_per_object: int | None = 4


@dataclasses.dataclass(kw_only=True)
class NASAImageLibraryCachedDatasetConfig(PagedApiCachedDatasetConfig, ImageDatasetConfig):
    api_url: str = "https://images-api.nasa.gov"
    site_url: str = "https://images.nasa.gov/details"
    metadata_file: str = "nasa-images-filtered.parquet"

    page_size: int = 100
    max_results_per_query: int = 10000
    min_request_interval: float = 0.25
    first_year: int = 1950
    last_year: int = 2026

    centers: list[str] | None = None
    split_centers: list[str] = dataclasses.field(default_factory=lambda: [
        "ARC", "AFRC", "GRC", "GSFC", "HQ", "JPL", "JSC", "KSC", "LRC", "LARC", "MSFC", "SSC", "WSTF",
    ])
    excluded_terms: list[str] = dataclasses.field(default_factory=lambda: [
        "logo", "logos", "logotype", "insignia", "meatball", "worm", "emblem", "seal", "patch", "patches", "decal",
        "©", "copyright", "copyrighted", "courtesy",
    ])
    asset_url: str = "https://images-assets.nasa.gov/image"
    size_preference: list[str] = dataclasses.field(default_factory=lambda: ["medium", "large", "orig", "small"])
    resolve_asset_sizes: bool = True
