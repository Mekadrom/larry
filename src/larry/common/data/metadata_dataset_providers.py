import functools
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import Dataset

from larry.common.config.data.preprocessing.dataset_configs import (
    FilteredMetadataCachedDatasetConfig,
    PagedApiCachedDatasetConfig,
)
from larry.common.data.cached_dataset_providers import CachedDatasetProvider
from larry.common.data.utils import bot_user_agent


class FilteredMetadataCachedDatasetProvider[C: FilteredMetadataCachedDatasetConfig = FilteredMetadataCachedDatasetConfig](
    CachedDatasetProvider[C], ABC
):
    ATTRIBUTION_FIELDS: ClassVar[list[pa.Field]] = [
        pa.field("id", pa.string()),
        pa.field("license", pa.string()),
        pa.field("license_raw", pa.string()),
        pa.field("creator", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("title", pa.string()),
        pa.field("description", pa.string()),
    ]
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = []

    @functools.cached_property
    def schema(self) -> pa.Schema:
        return pa.schema([*self.ATTRIBUTION_FIELDS, *self.EXTRA_FIELDS])

    @property
    def metadata_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.metadata_file

    @property
    def shard_dir(self) -> Path:
        return self.download_extract_cache_dir / "shards"

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return {"train": str(self.metadata_path)}

    @abstractmethod
    def units(self) -> Iterable[str]:
        ...

    @abstractmethod
    def rows(self, unit: str) -> Iterable[dict[str, Any]]:
        ...

    def after_shard(self, unit: str) -> None:
        ...

    def sampled(self, row_id: str) -> bool:
        if self.config.sample_fraction >= 1.0:
            return True

        bucket = int.from_bytes(hashlib.md5(row_id.encode("utf-8")).digest()[:8], "big") / 2 ** 64
        return bucket < self.config.sample_fraction

    def shard_path(self, unit: str) -> Path:
        return self.shard_dir / f"{unit}.parquet"

    def write_batch(self, writer: pq.ParquetWriter, batch: list[dict[str, Any]]) -> None:
        writer.write_table(pa.Table.from_pylist(batch, schema=self.schema))

    def write_shard(self, unit: str) -> int:
        shard = self.shard_path(unit)
        if shard.exists():
            return pq.ParquetFile(shard).metadata.num_rows

        shard.parent.mkdir(parents=True, exist_ok=True)
        partial = shard.with_name(f"{shard.name}.part")
        kept = 0
        batch: list[dict[str, Any]] = []
        with pq.ParquetWriter(partial, self.schema) as writer:
            for row in self.rows(unit):
                if not self.sampled(row["id"]):
                    continue

                batch.append(row)
                kept += 1
                if len(batch) >= self.config.write_batch_rows:
                    self.write_batch(writer, batch)
                    batch = []
                    self.log.info(f"{unit}: kept {kept} rows")

            if batch:
                self.write_batch(writer, batch)

        partial.rename(shard)
        self.after_shard(unit)
        return kept

    def shard_tables(self, unit: str) -> Iterator[pa.Table]:
        for batch in pq.ParquetFile(self.shard_path(unit)).iter_batches(self.config.write_batch_rows):
            yield pa.Table.from_batches([batch], schema=self.schema)

    def merge(self, units: list[str]) -> int:
        partial = self.metadata_path.with_name(f"{self.metadata_path.name}.part")
        seen: set[str | None] = set()
        merged = 0
        with pq.ParquetWriter(partial, self.schema) as writer:
            for unit in units:
                for table in self.shard_tables(unit):
                    if self.config.deduplicate_ids:
                        keep = []
                        for row_id in table.column("id").to_pylist():
                            keep.append(row_id not in seen)
                            seen.add(row_id)

                        table = table.filter(pa.array(keep, type=pa.bool_()))

                    writer.write_table(table)
                    merged += table.num_rows

        partial.rename(self.metadata_path)
        return merged

    def load_secondary_data(self) -> None:
        if self.metadata_path.exists():
            self.log.info(f"Already filtered: {self.metadata_path}")
            return

        units: list[str] = []
        kept = 0
        for unit in self.units():
            kept += self.write_shard(unit)
            units.append(unit)
            if len(units) % 100 == 0:
                self.log.info(f"filtered {len(units)} units, kept {kept} rows")

        merged = self.merge(units)
        self.log.info(f"kept {merged} rows from {len(units)} units")

    def cast_cols(self, dataset: Dataset) -> Dataset:
        return dataset

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return batch


class PagedApiCachedDatasetProvider[C: PagedApiCachedDatasetConfig = PagedApiCachedDatasetConfig](
    FilteredMetadataCachedDatasetProvider[C], ABC
):
    def __init__(self, config: C) -> None:
        super().__init__(config)
        self.last_request = 0.0

    @functools.cached_property
    def session(self) -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = bot_user_agent(self.config.user_agent_name)
        return session

    @property
    def api_key(self) -> str | None:
        name = self.config.api_key_env_var
        if name is None:
            return None

        key = os.environ.get(name)
        if not key:
            raise ValueError(f"environment variable {name} is not set")

        return key

    @property
    def cursors_path(self) -> Path:
        return self.download_extract_cache_dir / "cursors.json"

    @abstractmethod
    def request(self, cursor: str | None) -> tuple[list[Any], str | None]:
        ...

    @abstractmethod
    def rows_from_record(self, record: Any) -> Iterable[dict[str, Any]]:
        ...

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        response = self.session.get(url, params=params, timeout=self.config.request_timeout)
        response.raise_for_status()
        return response.json()

    def load_cursors(self) -> dict[str, str | None]:
        if not self.cursors_path.exists():
            return {}

        return json.loads(self.cursors_path.read_text())

    def save_cursor(self, page: int, cursor: str | None) -> None:
        cursors = self.load_cursors()
        cursors[str(page)] = cursor
        partial = self.cursors_path.with_name(f"{self.cursors_path.name}.part")
        partial.write_text(json.dumps(cursors))
        partial.rename(self.cursors_path)

    @staticmethod
    def unit_name(page: int) -> str:
        return f"page-{page:08d}"

    def units(self) -> Iterator[str]:
        page = 0
        while self.config.max_pages is None or page < self.config.max_pages:
            yield self.unit_name(page)
            cursors = self.load_cursors()
            if str(page + 1) not in cursors:
                raise RuntimeError(f"{self.unit_name(page)} has a shard but no saved cursor for the page after it")

            if cursors[str(page + 1)] is None:
                return

            page += 1

    def fetch(self, cursor: str | None) -> tuple[list[Any], str | None]:
        attempt = 0
        while True:
            wait = self.config.min_request_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)

            self.last_request = time.monotonic()
            try:
                return self.request(cursor)
            except (requests.RequestException, ValueError) as e:
                attempt += 1
                response = getattr(e, "response", None)
                reason = f"{type(e).__name__}, status {getattr(response, 'status_code', None)}"
                if attempt > self.config.max_retries:
                    raise RuntimeError(f"request failed after {attempt} attempts ({reason})") from None

                delay = self.config.retry_backoff * attempt
                self.log.warning(f"request failed ({reason}); retry {attempt} in {delay:.0f}s")
                time.sleep(delay)

    def rows(self, unit: str) -> Iterator[dict[str, Any]]:
        page = int(unit.removeprefix("page-"))
        cursor = self.load_cursors().get(str(page)) if page else None
        records, next_cursor = self.fetch(cursor)
        self.save_cursor(page + 1, next_cursor)
        for record in records:
            yield from self.rows_from_record(record)
