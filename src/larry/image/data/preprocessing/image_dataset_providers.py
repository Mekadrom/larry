import gzip
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from larry.common.data.cached_dataset_providers import CachedDatasetProvider, MediaCachedDatasetProvider
from larry.common.data.downloaders import SingleFileDownloader
from larry.image.config.preprocessing.image_dataset_configs import (
    GoogleDOCCIMediaCachedDatasetConfig,
    SmithsonianOpenAccessCachedDatasetConfig,
    WikimediaCommonsCachedDatasetConfig,
)


class GoogleDOCCIMediaCachedDatasetProvider(MediaCachedDatasetProvider[GoogleDOCCIMediaCachedDatasetConfig]):
    """Override for ``MediaCachedDatasetProvider`` to provide the google docci dataset as if it was an HF dataset.

    This in conjunction with the ``GoogleDOCCIMediaCachedDatasetConfig`` completely replaces the arbitrary python code
    execution required to load the original google/docci dataset straight from huggingface.
    """

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        """Returns the descriptions file, which is jsonlines, to start as the raw dataset."""
        return {"train": str(self.download_extract_cache_dir / self.config.descriptions_file)}


class WikimediaCommonsCachedDatasetProvider(CachedDatasetProvider[WikimediaCommonsCachedDatasetConfig]):
    STANDARD_THUMBNAIL_WIDTHS = (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840)
    URL_SAFE = ";@$!*(),~:"
    SCHEMA = pa.schema([
        ("id", pa.string()),
        ("page_id", pa.int64()),
        ("title", pa.string()),
        ("mime_type", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("license_ids", pa.list_(pa.string())),
        ("copyright_status_ids", pa.list_(pa.string())),
        ("copyright_determination_ids", pa.list_(pa.string())),
        ("source_ids", pa.list_(pa.string())),
        ("depicts_ids", pa.list_(pa.string())),
        ("caption_en", pa.string()),
        ("captions", pa.string()),
        ("image_url", pa.string()),
    ])

    def __init__(self, config: WikimediaCommonsCachedDatasetConfig) -> None:
        super().__init__(config)
        if config.thumbnail_width not in self.STANDARD_THUMBNAIL_WIDTHS:
            raise ValueError(
                f"thumbnail_width={config.thumbnail_width} is not a standard Wikimedia thumbnail width "
                f"{self.STANDARD_THUMBNAIL_WIDTHS}; upload.wikimedia.org rejects other sizes"
            )

        self.markers = [f'"{q}"'.encode() for q in (*config.licenses, *config.copyright_statuses)]

    @property
    def metadata_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.metadata_file

    @property
    def mediainfo_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.mediainfo_file

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return {"train": str(self.metadata_path)}

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
    def qualifier_ids(entity: dict[str, Any], prop: str, qualifier: str) -> list[str]:
        found = []
        for statement in entity.get("statements", {}).get(prop, []):
            for snak in statement.get("qualifiers", {}).get(qualifier, []):
                value = snak.get("datavalue", {}).get("value")
                if isinstance(value, dict) and "id" in value:
                    found.append(value["id"])

        return found

    def image_url(self, name: str, width: int | None) -> str:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        quoted = quote(name, safe=self.URL_SAFE)
        path = f"{digest[0]}/{digest[:2]}/{quoted}"
        thumbnail_width = self.config.thumbnail_width
        if width is not None and width > thumbnail_width:
            return f"{self.config.upload_url}/thumb/{path}/{thumbnail_width}px-{quoted}"

        return f"{self.config.upload_url}/{path}"

    def sampled(self, entity_id: str) -> bool:
        if self.config.sample_fraction >= 1.0:
            return True

        bucket = int.from_bytes(hashlib.md5(entity_id.encode("utf-8")).digest()[:8], "big") / 2 ** 64
        return bucket < self.config.sample_fraction

    def record(self, entity: dict[str, Any]) -> dict[str, Any] | None:
        if not self.sampled(entity["id"]):
            return None

        licenses = self.values(entity, "P275")
        statuses = self.values(entity, "P6216")
        if not set(licenses) & set(self.config.licenses) and not set(statuses) & set(self.config.copyright_statuses):
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

        title = entity.get("title", "")
        name = title.removeprefix("File:").replace(" ", "_")
        labels = {language: label["value"] for language, label in entity.get("labels", {}).items()}
        return {
            "id": entity["id"],
            "page_id": entity.get("pageid"),
            "title": name,
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "license_ids": licenses,
            "copyright_status_ids": statuses,
            "copyright_determination_ids": self.qualifier_ids(entity, "P6216", "P459"),
            "source_ids": self.values(entity, "P7482"),
            "depicts_ids": self.values(entity, "P180"),
            "caption_en": labels.get("en"),
            "captions": json.dumps(labels, ensure_ascii=False) if labels else None,
            "image_url": self.image_url(name, width),
        }

    def load_secondary_data(self) -> None:
        if self.metadata_path.exists():
            self.log.info(f"Already filtered: {self.metadata_path}")
            return

        partial = self.metadata_path.with_name(f"{self.metadata_path.name}.part")
        kept = scanned = 0
        batch: list[dict[str, Any]] = []
        with self.open_mediainfo() as stream, pq.ParquetWriter(partial, self.SCHEMA) as writer:
            for entity in self.candidate_entities(stream):
                scanned += 1
                row = self.record(entity)
                if row is None:
                    continue

                batch.append(row)
                kept += 1
                if len(batch) >= self.config.write_batch_rows:
                    writer.write_table(pa.Table.from_pylist(batch, schema=self.SCHEMA))
                    batch = []
                    self.log.info(f"kept {kept} of {scanned} candidate files")

            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=self.SCHEMA))

        self.log.info(f"kept {kept} of {scanned} candidate files")
        partial.rename(self.metadata_path)

    def cast_cols(self, dataset: Dataset) -> Dataset:
        return dataset

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return batch


class SmithsonianOpenAccessCachedDatasetProvider(CachedDatasetProvider[SmithsonianOpenAccessCachedDatasetConfig]):
    SCHEMA = pa.schema([
        ("id", pa.string()),
        ("record_id", pa.string()),
        ("record_guid", pa.string()),
        ("media_guid", pa.string()),
        ("unit_code", pa.string()),
        ("data_source", pa.string()),
        ("media_type", pa.string()),
        ("usage", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("title", pa.string()),
        ("media_caption", pa.string()),
        ("object_types", pa.list_(pa.string())),
        ("freetext", pa.string()),
        ("indexed_structured", pa.string()),
        ("image_url", pa.string()),
    ])

    @property
    def metadata_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.metadata_file

    @property
    def shard_dir(self) -> Path:
        return self.download_extract_cache_dir / "filtered"

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return {"train": str(self.metadata_path)}

    def relative_path(self, url: str) -> str:
        return url.strip().removeprefix(f"{self.config.bucket_url}/")

    def download_metadata(self, path: str) -> None:
        config = self.config.file_downloader_config(path)
        SingleFileDownloader(config, self.download_extract_cache_dir).ensure_downloaded()

    def read_index(self, path: str) -> list[str]:
        self.download_metadata(path)
        lines = (self.download_extract_cache_dir / path).read_text().splitlines()
        return [self.relative_path(line) for line in lines if line.strip()]

    def unit(self, path: str) -> str:
        return path.removeprefix(f"{self.config.metadata_prefix}/").split("/")[0]

    def selected(self, unit: str) -> bool:
        if unit in self.config.excluded_units:
            return False

        return self.config.units is None or unit in self.config.units

    def metadata_files(self) -> list[str]:
        files = []
        for unit_index in self.read_index(self.config.index_file):
            if self.selected(self.unit(unit_index)):
                files.extend(self.read_index(unit_index))

        return files

    @staticmethod
    def dimensions(media: dict[str, Any]) -> tuple[int | None, int | None]:
        if media.get("width") and media.get("height"):
            return int(media["width"]), int(media["height"])

        for resource in media.get("resources", []):
            width, _, height = (resource.get("dimensions") or "").partition("x")
            if width.isdigit() and height.isdigit():
                return int(width), int(height)

        return None, None

    def image_url(self, ids_id: str, width: int | None, height: int | None) -> str:
        url = f"{self.config.ids_url}?id={quote(ids_id, safe=':/')}"
        if width is not None and height is not None and max(width, height) <= self.config.max_size:
            return url

        return f"{url}&max={self.config.max_size}"

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
                "record_id": descriptive.get("record_ID"),
                "record_guid": descriptive.get("guid"),
                "media_guid": media.get("guid"),
                "unit_code": descriptive.get("unit_code") or entry.get("unitCode"),
                "data_source": descriptive.get("data_source"),
                "media_type": media["type"],
                "usage": usage,
                "width": width,
                "height": height,
                "title": descriptive.get("title", {}).get("content"),
                "media_caption": media.get("caption"),
                "object_types": indexed.get("object_type", []),
                "freetext": json.dumps(freetext, ensure_ascii=False) if freetext else None,
                "indexed_structured": json.dumps(indexed, ensure_ascii=False) if indexed else None,
                "image_url": self.image_url(ids_id, width, height),
            }

    def shard_path(self, path: str) -> Path:
        return self.shard_dir / Path(path).relative_to(self.config.metadata_prefix).with_suffix(".parquet")

    def filter_file(self, path: str) -> int:
        shard = self.shard_path(path)
        if shard.exists():
            return pq.ParquetFile(shard).metadata.num_rows

        self.download_metadata(path)
        raw = self.download_extract_cache_dir / path
        rows = []
        with raw.open("rb") as f:
            for line in f:
                if line.strip():
                    rows.extend(self.records(json.loads(line)))

        shard.parent.mkdir(parents=True, exist_ok=True)
        partial = shard.with_name(f"{shard.name}.part")
        pq.write_table(pa.Table.from_pylist(rows, schema=self.SCHEMA), partial)
        partial.rename(shard)
        if not self.config.keep_raw_metadata:
            raw.unlink()

        return len(rows)

    def load_secondary_data(self) -> None:
        if self.metadata_path.exists():
            self.log.info(f"Already filtered: {self.metadata_path}")
            return

        files = self.metadata_files()
        kept = 0
        for i, path in enumerate(files):
            kept += self.filter_file(path)
            if (i + 1) % 100 == 0:
                self.log.info(f"filtered {i + 1} of {len(files)} metadata files, kept {kept} images")

        partial = self.metadata_path.with_name(f"{self.metadata_path.name}.part")
        seen: set[str | None] = set()
        with pq.ParquetWriter(partial, self.SCHEMA) as writer:
            for path in files:
                table = pq.read_table(self.shard_path(path), schema=self.SCHEMA)
                ids = table.column("id").to_pylist()
                keep = [i not in seen for i in ids]
                seen.update(ids)
                writer.write_table(table.filter(pa.array(keep, type=pa.bool_())))

        self.log.info(f"kept {len(seen)} unique images from {len(files)} metadata files")
        partial.rename(self.metadata_path)

    def cast_cols(self, dataset: Dataset) -> Dataset:
        return dataset

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return batch
