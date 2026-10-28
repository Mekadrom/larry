import gzip
import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any, ClassVar
from urllib.parse import quote

import pyarrow as pa

from larry.common.data.cached_dataset_providers import MediaCachedDatasetProvider
from larry.common.data.downloaders import SingleFileDownloader
from larry.common.data.licenses import License
from larry.common.data.metadata_dataset_providers import FilteredMetadataCachedDatasetProvider
from larry.image.config.preprocessing.image_dataset_configs import (
    GoogleDOCCIMediaCachedDatasetConfig,
    SmithsonianOpenAccessCachedDatasetConfig,
    WikimediaCommonsCachedDatasetConfig,
)
from larry.image.data.preprocessing.image_urls import SmithsonianIdsUrls, WikimediaThumbnailUrls

IMAGE_FIELDS = [
    pa.field("image_url", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
]


class GoogleDOCCIMediaCachedDatasetProvider(MediaCachedDatasetProvider[GoogleDOCCIMediaCachedDatasetConfig]):
    """Override for ``MediaCachedDatasetProvider`` to provide the google docci dataset as if it was an HF dataset.

    This in conjunction with the ``GoogleDOCCIMediaCachedDatasetConfig`` completely replaces the arbitrary python code
    execution required to load the original google/docci dataset straight from huggingface.
    """

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        """Returns the descriptions file, which is jsonlines, to start as the raw dataset."""
        return {"train": str(self.download_extract_cache_dir / self.config.descriptions_file)}


class WikimediaCommonsCachedDatasetProvider(FilteredMetadataCachedDatasetProvider[WikimediaCommonsCachedDatasetConfig]):
    LICENSES: ClassVar[dict[str, License]] = {"Q6938433": License.CC0}
    COPYRIGHT_STATUSES: ClassVar[dict[str, License]] = {"Q19652": License.PUBLIC_DOMAIN}
    US_GOVERNMENT_DETERMINATION = "Q60671452"
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = [
        *IMAGE_FIELDS,
        pa.field("page_id", pa.int64()),
        pa.field("mime_type", pa.string()),
        pa.field("source_ids", pa.list_(pa.string())),
        pa.field("depicts_ids", pa.list_(pa.string())),
        pa.field("captions", pa.string()),
    ]

    def __init__(self, config: WikimediaCommonsCachedDatasetConfig) -> None:
        super().__init__(config)
        unmapped = [q for q in config.licenses if q not in self.LICENSES]
        unmapped += [q for q in config.copyright_statuses if q not in self.COPYRIGHT_STATUSES]
        if unmapped:
            raise ValueError(f"no canonical license is known for {unmapped}")

        self.urls = WikimediaThumbnailUrls(config.upload_url, config.thumbnail_width)
        self.markers = [f'"{q}"'.encode() for q in (*config.licenses, *config.copyright_statuses)]

    @property
    def mediainfo_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.mediainfo_file

    def open_mediainfo(self) -> IO[bytes] | gzip.GzipFile:
        if shutil.which("pigz"):
            process = subprocess.Popen(["pigz", "-dc", str(self.mediainfo_path)], stdout=subprocess.PIPE)
            assert process.stdout is not None
            return process.stdout

        return gzip.open(self.mediainfo_path, "rb")

    def candidate_entities(self, stream: Iterable[bytes]) -> Iterator[dict[str, Any]]:
        for line in stream:
            if not any(marker in line for marker in self.markers):
                continue

            line = line.strip().rstrip(b",")
            if line.startswith(b"{"):
                yield json.loads(line)

    @staticmethod
    def values(entity: dict[str, Any], prop: str) -> list[Any]:
        found = []
        for statement in entity.get("statements", {}).get(prop, []):
            datavalue = statement.get("mainsnak", {}).get("datavalue")
            if datavalue is None:
                continue

            value = datavalue["value"]
            if isinstance(value, dict) and "id" in value:
                found.append(value["id"])
            elif isinstance(value, dict) and "amount" in value:
                found.append(float(value["amount"]))
            else:
                found.append(value)

        return found

    @staticmethod
    def qualifier_values(entity: dict[str, Any], prop: str, qualifier: str) -> list[str]:
        found = []
        for statement in entity.get("statements", {}).get(prop, []):
            for snak in statement.get("qualifiers", {}).get(qualifier, []):
                value = snak.get("datavalue", {}).get("value")
                if isinstance(value, dict) and "id" in value:
                    found.append(value["id"])
                elif isinstance(value, str):
                    found.append(value)

        return found

    def creators(self, entity: dict[str, Any]) -> list[str]:
        return [
            *(str(v) for v in self.values(entity, "P170")),
            *self.qualifier_values(entity, "P170", "P2093"),
            *self.qualifier_values(entity, "P170", "P4174"),
        ]

    def license(self, licenses: list[str], statuses: list[str], determinations: list[str]) -> License | None:
        for q in licenses:
            if q in self.config.licenses:
                return self.LICENSES[q]

        for q in statuses:
            if q in self.config.copyright_statuses:
                if self.US_GOVERNMENT_DETERMINATION in determinations:
                    return License.US_GOVERNMENT

                return self.COPYRIGHT_STATUSES[q]

        return None

    def record(self, entity: dict[str, Any]) -> dict[str, Any] | None:
        licenses = self.values(entity, "P275")
        statuses = self.values(entity, "P6216")
        determinations = self.qualifier_values(entity, "P6216", "P459")
        lcns = self.license(licenses, statuses, determinations)
        if lcns is None:
            return None

        mime_types = self.values(entity, "P1163")
        mime_type = mime_types[0] if mime_types else None
        if mime_type not in self.config.mime_types:
            return None

        widths, heights = self.values(entity, "P2049"), self.values(entity, "P2048")
        width = int(widths[0]) if widths else None
        height = int(heights[0]) if heights else None
        if self.config.require_dimensions and (width is None or height is None):
            return None

        name = entity.get("title", "").removeprefix("File:").replace(" ", "_")
        labels = {language: label["value"] for language, label in entity.get("labels", {}).items()}
        creators = self.creators(entity)
        return {
            "id": entity["id"],
            "license": lcns,
            "license_raw": json.dumps({
                "licenses": licenses, "copyright_statuses": statuses, "copyright_determinations": determinations,
            }),
            "creator": "; ".join(creators) if creators else None,
            "source_url": f"{self.config.site_url}/File:{quote(name, safe=WikimediaThumbnailUrls.URL_SAFE)}",
            "title": name,
            "description": labels.get("en"),
            "image_url": self.urls.url(name, width, height),
            "width": width,
            "height": height,
            "page_id": entity.get("pageid"),
            "mime_type": mime_type,
            "source_ids": self.values(entity, "P7482"),
            "depicts_ids": self.values(entity, "P180"),
            "captions": json.dumps(labels, ensure_ascii=False) if labels else None,
        }

    def units(self) -> Iterable[str]:
        return ["mediainfo"]

    def rows(self, unit: str) -> Iterator[dict[str, Any]]:
        with self.open_mediainfo() as stream:
            for entity in self.candidate_entities(stream):
                row = self.record(entity)
                if row is not None:
                    yield row


class SmithsonianOpenAccessCachedDatasetProvider(
    FilteredMetadataCachedDatasetProvider[SmithsonianOpenAccessCachedDatasetConfig]
):
    USAGES: ClassVar[dict[str, License]] = {"CC0": License.CC0}
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = [
        *IMAGE_FIELDS,
        pa.field("record_id", pa.string()),
        pa.field("media_guid", pa.string()),
        pa.field("unit_code", pa.string()),
        pa.field("data_source", pa.string()),
        pa.field("media_type", pa.string()),
        pa.field("media_caption", pa.string()),
        pa.field("object_types", pa.list_(pa.string())),
        pa.field("freetext", pa.string()),
        pa.field("indexed_structured", pa.string()),
    ]

    def __init__(self, config: SmithsonianOpenAccessCachedDatasetConfig) -> None:
        super().__init__(config)
        unmapped = [usage for usage in config.usages if usage not in self.USAGES]
        if unmapped:
            raise ValueError(f"no canonical license is known for {unmapped}")

        self.urls = SmithsonianIdsUrls(config.ids_url, config.max_size)

    def relative_path(self, url: str) -> str:
        return url.strip().removeprefix(f"{self.config.bucket_url}/")

    def download_metadata(self, path: str) -> None:
        config = self.config.file_downloader_config(path)
        SingleFileDownloader(config, self.download_extract_cache_dir).ensure_downloaded()

    def read_index(self, path: str) -> list[str]:
        self.download_metadata(path)
        lines = (self.download_extract_cache_dir / path).read_text().splitlines()
        return [self.relative_path(line) for line in lines if line.strip()]

    def unit_of(self, path: str) -> str:
        return path.removeprefix(f"{self.config.metadata_prefix}/").split("/")[0]

    def selected(self, unit: str) -> bool:
        if unit in self.config.excluded_units:
            return False

        return self.config.units is None or unit in self.config.units

    def metadata_path_of(self, unit: str) -> str:
        return f"{self.config.metadata_prefix}/{unit}.txt"

    def units(self) -> Iterator[str]:
        for unit_index in self.read_index(self.config.index_file):
            if not self.selected(self.unit_of(unit_index)):
                continue

            for path in self.read_index(unit_index):
                yield path.removeprefix(f"{self.config.metadata_prefix}/").removesuffix(".txt")

    @staticmethod
    def dimensions(media: dict[str, Any]) -> tuple[int | None, int | None]:
        if media.get("width") and media.get("height"):
            return int(media["width"]), int(media["height"])

        for resource in media.get("resources", []):
            width, _, height = (resource.get("dimensions") or "").partition("x")
            if width.isdigit() and height.isdigit():
                return int(width), int(height)

        return None, None

    def description(self, freetext: dict[str, Any]) -> str | None:
        notes = [n["content"] for n in freetext.get("notes", []) if n.get("label") in self.config.description_labels]
        return "\n\n".join(notes) if notes else None

    @staticmethod
    def creator(freetext: dict[str, Any]) -> str | None:
        names = [f"{n['label']}: {n['content']}" for n in freetext.get("name", []) if n.get("content")]
        return "; ".join(names) if names else None

    def records(self, entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
        content = entry.get("content", {})
        descriptive = content.get("descriptiveNonRepeating", {})
        indexed = content.get("indexedStructured", {})
        freetext = content.get("freetext", {})
        for media in descriptive.get("online_media", {}).get("media", []):
            ids_id = media.get("idsId")
            usage = media.get("usage", {}).get("access")
            if ids_id is None or media.get("type") not in self.config.media_types or usage not in self.config.usages:
                continue

            width, height = self.dimensions(media)
            yield {
                "id": ids_id,
                "license": self.USAGES[usage],
                "license_raw": usage,
                "creator": self.creator(freetext),
                "source_url": descriptive.get("guid"),
                "title": descriptive.get("title", {}).get("content"),
                "description": self.description(freetext),
                "image_url": self.urls.url(ids_id, width, height),
                "width": width,
                "height": height,
                "record_id": descriptive.get("record_ID"),
                "media_guid": media.get("guid"),
                "unit_code": descriptive.get("unit_code") or entry.get("unitCode"),
                "data_source": descriptive.get("data_source"),
                "media_type": media["type"],
                "media_caption": media.get("caption"),
                "object_types": indexed.get("object_type", []),
                "freetext": json.dumps(freetext, ensure_ascii=False) if freetext else None,
                "indexed_structured": json.dumps(indexed, ensure_ascii=False) if indexed else None,
            }

    def rows(self, unit: str) -> Iterator[dict[str, Any]]:
        path = self.metadata_path_of(unit)
        self.download_metadata(path)
        with (self.download_extract_cache_dir / path).open("rb") as f:
            for line in f:
                if line.strip():
                    yield from self.records(json.loads(line))

    def after_shard(self, unit: str) -> None:
        if not self.config.keep_raw_metadata:
            (self.download_extract_cache_dir / self.metadata_path_of(unit)).unlink()
