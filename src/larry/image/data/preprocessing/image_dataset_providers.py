import gzip
import html
import json
import re
import shutil
import subprocess
import tarfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any, ClassVar
from urllib.parse import quote

import duckdb
import pyarrow as pa

from larry.common.data.cached_dataset_providers import MediaCachedDatasetProvider
from larry.common.data.downloaders import SingleFileDownloader
from larry.common.data.licenses import License
from larry.common.data.metadata_dataset_providers import (
    FilteredMetadataCachedDatasetProvider,
    PagedApiCachedDatasetProvider,
)
from larry.common.data.term_matching import TermMatcher
from larry.image.config.preprocessing.image_dataset_configs import (
    ArtInstituteOfChicagoCachedDatasetConfig,
    GoogleDOCCIMediaCachedDatasetConfig,
    NASAImageLibraryCachedDatasetConfig,
    SmithsonianOpenAccessCachedDatasetConfig,
    TePapaCachedDatasetConfig,
    WikimediaCommonsCachedDatasetConfig,
)
from larry.image.data.preprocessing.image_urls import IIIFImageUrls, SmithsonianIdsUrls, WikimediaThumbnailUrls

TAG = re.compile(r"<[^>]+>")


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None

    cleaned = " ".join(TAG.sub("", html.unescape(text)).split())
    return cleaned or None


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
        self.require_canonical_licenses(config.licenses, self.LICENSES)
        self.require_canonical_licenses(config.copyright_statuses, self.COPYRIGHT_STATUSES)

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
        pa.field("texts", pa.list_(pa.struct([
            ("field", pa.string()), ("label", pa.string()), ("content", pa.string()),
        ]))),
        pa.field("freetext", pa.string()),
        pa.field("indexed_structured", pa.string()),
    ]

    def __init__(self, config: SmithsonianOpenAccessCachedDatasetConfig) -> None:
        super().__init__(config)
        self.require_canonical_licenses(config.usages, self.USAGES)

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
        notes = [
            clean_text(n.get("content"))
            for n in freetext.get("notes", [])
            if n.get("label") in self.config.description_labels
        ]
        notes = [n for n in notes if n]
        return "\n\n".join(notes) if notes else None

    def texts(self, freetext: dict[str, Any]) -> list[dict[str, str | None]]:
        found = []
        for field, entries in freetext.items():
            for entry in entries if isinstance(entries, list) else []:
                content = clean_text(entry.get("content")) if isinstance(entry, dict) else None
                if content:
                    found.append({"field": field, "label": clean_text(entry.get("label")), "content": content})

        return found

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
                "title": clean_text(descriptive.get("title", {}).get("content")),
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
                "texts": self.texts(freetext),
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

    def shard_query(self, unit: str) -> str | None:
        where = self.config.unit_filters.get(unit.split("/")[0])
        if where is None:
            return None

        path = str(self.shard_path(unit)).replace("'", "''")
        return f"""
            SELECT * EXCLUDE (record_media_count) FROM (
                SELECT *, count(*) OVER (PARTITION BY record_id) AS record_media_count
                FROM read_parquet('{path}')
            ) WHERE {where}
        """

    def filtered_shard_tables(self, unit: str) -> Iterator[pa.Table]:
        query = self.shard_query(unit)
        if query is None:
            yield from super().shard_tables(unit)
            return

        yield self.conform(duckdb.sql(query).to_arrow_table())

    def with_text(self, table: pa.Table) -> pa.Table:
        freetexts = [json.loads(f) if f else {} for f in table.column("freetext").to_pylist()]
        columns = {
            "title": [clean_text(t) for t in table.column("title").to_pylist()],
            "description": [self.description(f) for f in freetexts],
            "texts": [self.texts(f) for f in freetexts],
        }
        for name, values in columns.items():
            field = self.schema.field(name)
            table = table.set_column(self.schema.get_field_index(name), field, pa.array(values, field.type))

        return table

    def shard_tables(self, unit: str) -> Iterator[pa.Table]:
        reason = self.config.generation_excluded_units.get(unit.split("/")[0])
        for table in self.filtered_shard_tables(unit):
            yield self.add_generation_exclusions(self.with_text(table), [reason] if reason else [])

    def after_shard(self, unit: str) -> None:
        if not self.config.keep_raw_metadata:
            (self.download_extract_cache_dir / self.metadata_path_of(unit)).unlink()


class ArtInstituteOfChicagoCachedDatasetProvider(
    FilteredMetadataCachedDatasetProvider[ArtInstituteOfChicagoCachedDatasetConfig]
):
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = [
        *IMAGE_FIELDS,
        pa.field("artwork_id", pa.int64()),
        pa.field("is_alt_image", pa.bool_()),
        pa.field("artwork_type", pa.string()),
        pa.field("medium", pa.string()),
        pa.field("alt_text", pa.string()),
        pa.field("short_description", pa.string()),
        pa.field("date_display", pa.string()),
        pa.field("place_of_origin", pa.string()),
        pa.field("department", pa.string()),
        pa.field("classification_titles", pa.list_(pa.string())),
        pa.field("subject_titles", pa.list_(pa.string())),
        pa.field("term_titles", pa.list_(pa.string())),
    ]

    def __init__(self, config: ArtInstituteOfChicagoCachedDatasetConfig) -> None:
        super().__init__(config)
        self.urls = IIIFImageUrls(
            config.image_width, base_url=config.iiif_url, full_size="full", size_template="{size},"
        )

    @property
    def dump_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.dump_file

    def units(self) -> Iterable[str]:
        return ["artworks"]

    def artworks(self) -> Iterator[dict[str, Any]]:
        with tarfile.open(self.dump_path, "r:bz2") as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith(self.config.artworks_prefix):
                    continue

                if not member.name.endswith(".json"):
                    continue

                handle = archive.extractfile(member)
                if handle is not None:
                    yield json.load(handle)

    def records(self, artwork: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if not artwork.get("is_public_domain") or not artwork.get("image_id"):
            return

        thumbnail = artwork.get("thumbnail") or {}
        image_ids = [(artwork["image_id"], False)]
        if self.config.include_alt_images:
            image_ids += [(i, True) for i in artwork.get("alt_image_ids") or []]

        for image_id, is_alt in image_ids:
            width = None if is_alt else thumbnail.get("width")
            height = None if is_alt else thumbnail.get("height")
            yield {
                "id": image_id,
                "license": License.PUBLIC_DOMAIN,
                "license_raw": "is_public_domain",
                "creator": artwork.get("artist_display"),
                "source_url": f"{self.config.site_url}/{artwork['id']}",
                "title": artwork.get("title"),
                "description": clean_text(artwork.get("description")),
                "image_url": self.urls.url(image_id, width, height),
                "width": width,
                "height": height,
                "artwork_id": artwork["id"],
                "is_alt_image": is_alt,
                "artwork_type": artwork.get("artwork_type_title"),
                "medium": artwork.get("medium_display"),
                "alt_text": None if is_alt else thumbnail.get("alt_text"),
                "short_description": clean_text(artwork.get("short_description")),
                "date_display": artwork.get("date_display"),
                "place_of_origin": artwork.get("place_of_origin"),
                "department": artwork.get("department_title"),
                "classification_titles": artwork.get("classification_titles") or [],
                "subject_titles": artwork.get("subject_titles") or [],
                "term_titles": artwork.get("term_titles") or [],
            }

    def rows(self, unit: str) -> Iterator[dict[str, Any]]:
        for artwork in self.artworks():
            yield from self.records(artwork)


class TePapaCachedDatasetProvider(PagedApiCachedDatasetProvider[TePapaCachedDatasetConfig]):
    LICENSES: ClassVar[dict[str, License]] = {
        "No Known Copyright Restrictions": License.PUBLIC_DOMAIN,
        "CC BY 4.0": License.CC_BY,
        "CC 0": License.CC0,
    }
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = [
        *IMAGE_FIELDS,
        pa.field("object_id", pa.int64()),
        pa.field("collection", pa.string()),
        pa.field("credit_caption", pa.string()),
        pa.field("credit_line", pa.string()),
        pa.field("made_of", pa.string()),
        pa.field("categories", pa.list_(pa.string())),
        pa.field("object_types", pa.list_(pa.string())),
        pa.field("production", pa.list_(pa.string())),
    ]

    def __init__(self, config: TePapaCachedDatasetConfig) -> None:
        super().__init__(config)
        self.require_canonical_licenses(config.licenses, self.LICENSES)
        self.generation_terms = {reason: TermMatcher(terms) for reason, terms in config.generation_excluded_terms.items()}

    @property
    def query(self) -> str:
        licenses = " OR ".join(f'hasRepresentation.rights.title:"{title}"' for title in self.config.licenses)
        return f"type:Object AND ({licenses})"

    def request(self, unit: str, cursor: str | None) -> tuple[list[Any], str | None]:
        headers = {"x-api-key": self.api_key or "", "Accept": "application/json"}
        if cursor is None:
            response = self.session.post(
                f"{self.config.api_url}/search/_scroll",
                params={
                    "q": self.query,
                    "size": self.config.page_size,
                    "duration": self.config.scroll_minutes,
                    "fields": ",".join(self.config.fields),
                },
                headers=headers,
                allow_redirects=False,
                timeout=self.config.request_timeout,
            )
        else:
            response = self.session.get(
                cursor, headers=headers, allow_redirects=False, timeout=self.config.request_timeout
            )

        if response.status_code == 204:
            return [], None

        response.raise_for_status()
        location = response.headers.get("Location")
        records = response.json().get("results", [])
        return records, self.scroll_url(location) if location and records else None

    def scroll_url(self, location: str) -> str:
        if location.startswith(("http://", "https://")):
            return location

        return f"{self.config.api_url.rstrip('/')}/{location.lstrip('/')}"

    @staticmethod
    def titles(entries: Any) -> list[str]:
        return [e["title"] for e in entries or [] if isinstance(e, dict) and e.get("title")]

    def generation_exclusion_reasons(self, record: dict[str, Any]) -> list[str]:
        reasons = []
        collection = record.get("collection")
        if collection in self.config.generation_excluded_collections:
            reasons.append(f"collection:{collection}")

        texts = [record.get("title"), record.get("description"), record.get("caption")]
        texts += self.titles(record.get("depicts")) + self.titles(record.get("refersTo"))
        reasons += [reason for reason, matcher in self.generation_terms.items() if matcher.matches(texts)]
        return reasons

    def rows_from_record(self, record: Any) -> Iterable[dict[str, Any]]:
        if record.get("type") != "Object" or record.get("collection") in self.config.excluded_collections:
            return

        reasons = self.generation_exclusion_reasons(record)
        production = record.get("production") or []
        creators = [p["contributor"]["title"] for p in production if (p.get("contributor") or {}).get("title")]
        usable = [
            media for media in record.get("hasRepresentation") or []
            if media.get("type") == "ImageObject"
            and (media.get("rights") or {}).get("title") in self.config.licenses
            and (media.get("rights") or {}).get("allowsDownload")
            and media.get("previewUrl")
        ]
        for media in usable[:self.config.max_images_per_object]:
            rights = media["rights"]

            yield {
                "id": str(media["id"]),
                "generation_exclusion_reasons": list(reasons),
                "license": self.LICENSES[rights["title"]],
                "license_raw": rights["title"],
                "creator": "; ".join(dict.fromkeys(creators)) or None,
                "source_url": f"{self.config.site_url}/{record['id']}",
                "title": clean_text(record.get("title")),
                "description": clean_text(record.get("description")),
                "image_url": media["previewUrl"],
                "width": media.get("width"),
                "height": media.get("height"),
                "object_id": int(record["id"]),
                "collection": record.get("collection"),
                "credit_caption": clean_text(record.get("caption")),
                "credit_line": clean_text(record.get("creditLine")),
                "made_of": clean_text(record.get("isMadeOfSummary")),
                "categories": list(dict.fromkeys(
                    self.titles(record.get("depicts")) + self.titles(record.get("refersTo"))
                )),
                "object_types": self.titles(record.get("isTypeOf")),
                "production": self.titles(production),
            }


class NASAImageLibraryCachedDatasetProvider(PagedApiCachedDatasetProvider[NASAImageLibraryCachedDatasetConfig]):
    LICENSE_RAW = "NASA media usage guidelines"
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = [
        *IMAGE_FIELDS,
        pa.field("center", pa.string()),
        pa.field("date_created", pa.string()),
        pa.field("keywords", pa.list_(pa.string())),
        pa.field("photographer", pa.string()),
        pa.field("secondary_creator", pa.string()),
        pa.field("location", pa.string()),
    ]

    def __init__(self, config: NASAImageLibraryCachedDatasetConfig) -> None:
        super().__init__(config)
        self.excluded_terms = TermMatcher(config.excluded_terms)

    @staticmethod
    def unit_name(start: int, end: int, center: str | None = None) -> str:
        years = f"{start}_{end}"
        return years if center is None else f"{years}/{center}"

    @staticmethod
    def parse_unit(unit: str) -> dict[str, Any]:
        years, _, center = unit.partition("/")
        start, end = years.split("_")
        return {"year_start": int(start), "year_end": int(end), **({"center": center} if center else {})}

    def search_params(self, unit: str, page: int, page_size: int) -> dict[str, Any]:
        return {"media_type": "image", "page": page, "page_size": page_size, **self.parse_unit(unit)}

    def hits(self, unit: str) -> int:
        response = self.call(lambda: self.get_json(f"{self.config.api_url}/search", self.search_params(unit, 1, 1)))
        return int(response["collection"]["metadata"]["total_hits"])

    def period_units(self, start: int, end: int) -> list[str]:
        if self.config.centers is not None:
            return [self.unit_name(start, end, c) for c in self.config.centers]

        unit = self.unit_name(start, end)
        total = self.hits(unit)
        if total <= self.config.max_results_per_query:
            return [unit] if total else []

        split = [self.unit_name(start, end, c) for c in self.config.split_centers]
        covered = sum(self.hits(u) for u in split)
        if covered < total:
            self.log.warning(f"{unit}: {total - covered} of {total} images are under centers not in split_centers")

        return split

    def units(self) -> Iterator[str]:
        yield from self.period_units(1, self.config.first_year - 1)
        for year in range(self.config.first_year, self.config.last_year + 1):
            yield from self.period_units(year, year)

    def request(self, unit: str, cursor: str | None) -> tuple[list[Any], str | None]:
        page = int(cursor or 1)
        params = self.search_params(unit, page, self.config.page_size)
        collection = self.get_json(f"{self.config.api_url}/search", params)["collection"]
        has_next = any(link.get("rel") == "next" for link in collection.get("links", []))
        last_page = self.config.max_results_per_query // self.config.page_size
        return collection.get("items", []), str(page + 1) if has_next and page < last_page else None

    @staticmethod
    def as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]

        return [str(v) for v in value or []]

    @staticmethod
    def variants(urls: Iterable[str]) -> dict[str, str]:
        found = {}
        for url in urls:
            stem, _, extension = url.rpartition(".")
            if extension.lower() in ("jpg", "jpeg") and "~" in stem:
                found.setdefault(stem.rpartition("~")[2], url.replace("http://", "https://", 1))

        return found

    def listed_assets(self, nasa_id: str) -> list[str]:
        url = f"{self.config.asset_url}/{quote(nasa_id, safe='')}/collection.json"
        return self.call(lambda: self.get_json(url))

    def image_url(self, nasa_id: str, links: list[dict[str, Any]]) -> str | None:
        variants = self.variants(link.get("href", "") for link in links if link.get("render") == "image")
        preferred = self.config.size_preference[:2]
        if self.config.resolve_asset_sizes and not any(name in variants for name in preferred):
            variants = self.variants(self.listed_assets(nasa_id)) or variants

        return next((variants[name] for name in self.config.size_preference if name in variants), None)

    def rows_from_record(self, record: Any) -> Iterable[dict[str, Any]]:
        data = (record.get("data") or [{}])[0]
        if data.get("media_type") != "image" or not data.get("nasa_id"):
            return

        keywords = self.as_list(data.get("keywords"))
        texts = [data.get("title"), data.get("description"), data.get("photographer"), data.get("secondary_creator")]
        if self.excluded_terms.matches(texts + keywords):
            return

        links = record.get("links") or []
        nasa_id = data["nasa_id"]
        image_url = self.image_url(nasa_id, links)
        if image_url is None:
            return

        canonical = next((li for li in links if li.get("rel") == "canonical"), {})
        yield {
            "id": nasa_id,
            "license": License.US_GOVERNMENT,
            "license_raw": self.LICENSE_RAW,
            "creator": data.get("secondary_creator") or data.get("photographer") or data.get("center"),
            "source_url": f"{self.config.site_url}/{quote(nasa_id, safe='')}",
            "title": clean_text(data.get("title")),
            "description": clean_text(data.get("description") or data.get("description_508")),
            "image_url": image_url,
            "width": canonical.get("width"),
            "height": canonical.get("height"),
            "center": data.get("center"),
            "date_created": data.get("date_created"),
            "keywords": keywords,
            "photographer": data.get("photographer"),
            "secondary_creator": data.get("secondary_creator"),
            "location": data.get("location"),
        }
