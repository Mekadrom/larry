import functools
import hashlib
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
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
from larry.common.data.licenses import License
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
    GENERATION_FIELDS: ClassVar[list[pa.Field]] = [
        pa.field("generation_excluded", pa.bool_()),
        pa.field("generation_exclusion_reasons", pa.list_(pa.string())),
    ]
    EXTRA_FIELDS: ClassVar[list[pa.Field]] = []

    @functools.cached_property
    def schema(self) -> pa.Schema:
        return pa.schema([*self.ATTRIBUTION_FIELDS, *self.GENERATION_FIELDS, *self.EXTRA_FIELDS])

    @property
    def metadata_path(self) -> Path:
        return self.download_extract_cache_dir / self.config.metadata_file

    @property
    def shard_dir(self) -> Path:
        if self.config.shards_dir is not None:
            return Path(self.config.shards_dir).expanduser().resolve()

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

    @staticmethod
    def require_canonical_licenses(configured: Iterable[str], known: Mapping[str, License]) -> None:
        unmapped = [value for value in configured if value not in known]
        if unmapped:
            raise ValueError(f"no canonical license is known for {unmapped}")

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

                reasons = row.setdefault("generation_exclusion_reasons", [])
                row["generation_excluded"] = bool(reasons)
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

    def conform(self, table: pa.Table) -> pa.Table:
        for field in self.schema:
            if field.name not in table.column_names:
                table = table.append_column(field, pa.nulls(table.num_rows, field.type))

        table = table.select(self.schema.names).cast(self.schema)
        reasons = table.column("generation_exclusion_reasons").to_pylist()
        reasons = [r or [] for r in reasons]
        table = self.set_column(table, "generation_exclusion_reasons", reasons)
        return self.set_column(table, "generation_excluded", [bool(r) for r in reasons])

    def set_column(self, table: pa.Table, name: str, values: list[Any]) -> pa.Table:
        field = self.schema.field(name)
        return table.set_column(self.schema.get_field_index(name), field, pa.array(values, field.type))

    def add_generation_exclusions(self, table: pa.Table, reasons: list[str]) -> pa.Table:
        if not reasons:
            return table

        existing = table.column("generation_exclusion_reasons").to_pylist()
        merged = [list(dict.fromkeys([*(r or []), *reasons])) for r in existing]
        table = self.set_column(table, "generation_exclusion_reasons", merged)
        return self.set_column(table, "generation_excluded", [True] * table.num_rows)

    def shard_tables(self, unit: str) -> Iterator[pa.Table]:
        for batch in pq.ParquetFile(self.shard_path(unit)).iter_batches(self.config.write_batch_rows):
            yield self.conform(pa.Table.from_batches([batch]))

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

    @functools.cached_property
    def api_key(self) -> str | None:
        name, path = self.config.api_key_env_var, self.config.api_key_file
        if name is None and path is None:
            return None

        key = os.environ.get(name) if name is not None else None
        if not key and path is not None and Path(path).expanduser().is_file():
            key = Path(path).expanduser().read_text().strip()

        if not key:
            raise ValueError(f"no API key in environment variable {name} or file {path}")

        return key

    @abstractmethod
    def request(self, unit: str, cursor: str | None) -> tuple[list[Any], str | None]:
        ...

    @abstractmethod
    def rows_from_record(self, record: Any) -> Iterable[dict[str, Any]]:
        ...

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        response = self.session.get(url, params=params, timeout=self.config.request_timeout)
        response.raise_for_status()
        return response.json()

    def units(self) -> Iterable[str]:
        return ["all"]

    def call[T](self, request: Callable[[], T]) -> T:
        attempt = 0
        while True:
            wait = self.config.min_request_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)

            self.last_request = time.monotonic()
            try:
                return request()
            except (requests.RequestException, ValueError) as e:
                attempt += 1
                response = getattr(e, "response", None)
                reason = f"{type(e).__name__}, status {getattr(response, 'status_code', None)}"
                if attempt > self.config.max_retries:
                    raise RuntimeError(f"request failed after {attempt} attempts ({reason})") from None

                delay = self.config.retry_backoff * attempt
                self.log.warning(f"request failed ({reason}); retry {attempt} in {delay:.0f}s")
                time.sleep(delay)

    def fetch(self, unit: str, cursor: str | None) -> tuple[list[Any], str | None]:
        return self.call(lambda: self.request(unit, cursor))

    def rows(self, unit: str) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        pages = 0
        while True:
            records, cursor = self.fetch(unit, cursor)
            pages += 1
            for record in records:
                yield from self.rows_from_record(record)

            if cursor is None or (self.config.max_pages is not None and pages >= self.config.max_pages):
                return
