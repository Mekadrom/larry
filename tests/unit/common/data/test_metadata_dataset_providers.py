import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import requests

from larry.common.config.data.downloader_configs import DownloaderConfig
from larry.common.config.data.extractor_configs import ExtractorConfig
from larry.common.config.data.preprocessing.dataset_configs import (
    FilteredMetadataCachedDatasetConfig,
    PagedApiCachedDatasetConfig,
)
from larry.common.data.metadata_dataset_providers import (
    FilteredMetadataCachedDatasetProvider,
    PagedApiCachedDatasetProvider,
)


@dataclasses.dataclass(kw_only=True)
class UnitsConfig(FilteredMetadataCachedDatasetConfig):
    metadata_file: str = "filtered.parquet"
    write_batch_rows: int = 2

    def downloader_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return []

    def extractor_configs(self) -> Sequence[tuple[str, ExtractorConfig]]:
        return []


class UnitsProvider(FilteredMetadataCachedDatasetProvider[UnitsConfig], register=False):
    UNITS = {
        "a/0": ["x1", "x2", "x3"],
        "a/1": ["x3", "x4", "x4"],
        "b/0": [],
    }

    def __init__(self, config: UnitsConfig) -> None:
        super().__init__(config)
        self.requested: list[str] = []
        self.finished: list[str] = []

    def units(self) -> Iterable[str]:
        return list(self.UNITS)

    def rows(self, unit: str) -> Iterable[dict[str, Any]]:
        self.requested.append(unit)
        for row_id in self.UNITS[unit]:
            yield {"id": row_id, "license": "CC0", "title": f"{unit}:{row_id}"}

    def after_shard(self, unit: str) -> None:
        self.finished.append(unit)


def _units(tmp_path: Path, **overrides: Any) -> UnitsProvider:
    return UnitsProvider(UnitsConfig(**{
        "download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), **overrides,
    }))


def _read(provider: FilteredMetadataCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_units_are_merged_in_order_with_repeated_ids_dropped(tmp_path: Path) -> None:
    rows = _read(_units(tmp_path))

    assert [r["id"] for r in rows] == ["x1", "x2", "x3", "x4"]
    assert rows[2]["title"] == "a/0:x3"


def test_every_row_has_the_attribution_columns(tmp_path: Path) -> None:
    provider = _units(tmp_path)

    assert {"id", "license", "license_raw", "creator", "source_url", "title", "description"} <= set(
        provider.schema.names
    )
    assert _read(provider)[0]["creator"] is None


def test_deduplication_can_be_turned_off(tmp_path: Path) -> None:
    rows = _read(_units(tmp_path, deduplicate_ids=False))

    assert [r["id"] for r in rows] == ["x1", "x2", "x3", "x3", "x4", "x4"]


def test_finished_units_are_not_redone(tmp_path: Path) -> None:
    first = _units(tmp_path)
    rows = _read(first)
    first.metadata_path.unlink()
    second = _units(tmp_path)

    assert _read(second) == rows
    assert first.finished == list(UnitsProvider.UNITS)
    assert second.requested == [] and second.finished == []


def test_a_filtered_file_is_reused(tmp_path: Path) -> None:
    _read(_units(tmp_path))
    again = _units(tmp_path)
    again.load_secondary_data()

    assert again.requested == []


def test_sampling_is_a_deterministic_subset_of_the_same_size(tmp_path: Path) -> None:
    provider = _units(tmp_path, sample_fraction=0.25)
    ids = [f"M{i}" for i in range(20000)]

    first = [i for i in ids if provider.sampled(i)]

    assert first == [i for i in ids if provider.sampled(i)]
    assert 4600 < len(first) < 5400


def test_growing_the_sample_keeps_everything_already_sampled(tmp_path: Path) -> None:
    small, large = _units(tmp_path, sample_fraction=0.1), _units(tmp_path, sample_fraction=0.3)
    ids = [f"M{i}" for i in range(5000)]

    assert {i for i in ids if small.sampled(i)} <= {i for i in ids if large.sampled(i)}


@dataclasses.dataclass(kw_only=True)
class PagesConfig(PagedApiCachedDatasetConfig):
    metadata_file: str = "filtered.parquet"
    api_url: str = "https://example.org/api"
    min_request_interval: float = 0.0
    retry_backoff: float = 0.0


class PagesProvider(PagedApiCachedDatasetProvider[PagesConfig], register=False):
    PAGES = {None: (["a", "b"], "c2"), "c2": (["c"], "c3"), "c3": (["d", "a"], None)}

    def __init__(self, config: PagesConfig, failures: int = 0, fail_on: str | None = "c2") -> None:
        super().__init__(config)
        self.failures = failures
        self.fail_on = fail_on
        self.cursors: list[str | None] = []

    def request(self, cursor: str | None) -> tuple[list[Any], str | None]:
        self.cursors.append(cursor)
        if cursor == self.fail_on and self.failures:
            self.failures -= 1
            raise requests.ConnectionError("https://example.org/api?key=secret")

        records, next_cursor = self.PAGES[cursor]
        return [{"key": r} for r in records], next_cursor

    def rows_from_record(self, record: Any) -> Iterable[dict[str, Any]]:
        yield {"id": record["key"], "license": "CC0"}


def _pages(tmp_path: Path, **kwargs: Any) -> PagesProvider:
    overrides = {k: kwargs.pop(k) for k in list(kwargs) if k not in ("failures", "fail_on")}
    config = PagesConfig(**{"download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), **overrides})
    return PagesProvider(config, **kwargs)


def test_pages_are_followed_until_the_cursor_runs_out(tmp_path: Path) -> None:
    provider = _pages(tmp_path)

    assert [r["id"] for r in _read(provider)] == ["a", "b", "c", "d"]
    assert provider.cursors == [None, "c2", "c3"]


def test_a_stopped_run_resumes_from_the_saved_cursor(tmp_path: Path) -> None:
    first = _pages(tmp_path, max_pages=2)
    _read(first)
    first.metadata_path.unlink()
    second = _pages(tmp_path)

    assert [r["id"] for r in _read(second)] == ["a", "b", "c", "d"]
    assert second.cursors == ["c3"]


def test_failed_requests_are_retried(tmp_path: Path) -> None:
    provider = _pages(tmp_path, failures=2)

    assert [r["id"] for r in _read(provider)] == ["a", "b", "c", "d"]
    assert provider.cursors == [None, "c2", "c2", "c2", "c3"]


def test_giving_up_does_not_leak_the_request_url(tmp_path: Path) -> None:
    provider = _pages(tmp_path, failures=10, max_retries=1)

    with pytest.raises(RuntimeError) as raised:
        provider.load_secondary_data()

    assert "secret" not in str(raised.value) and raised.value.__cause__ is None
    assert "ConnectionError" in str(raised.value)


def test_api_keys_come_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _pages(tmp_path, api_key_env_var="LARRY_TEST_API_KEY")
    monkeypatch.delenv("LARRY_TEST_API_KEY", raising=False)

    with pytest.raises(ValueError, match="no API key in environment variable LARRY_TEST_API_KEY"):
        _ = provider.api_key

    monkeypatch.setenv("LARRY_TEST_API_KEY", "abc")
    assert provider.api_key == "abc"


def test_api_keys_can_come_from_a_file_when_the_environment_has_none(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("from-file\n")
    monkeypatch.delenv("LARRY_TEST_API_KEY", raising=False)

    assert _pages(tmp_path, api_key_env_var="LARRY_TEST_API_KEY", api_key_file=str(key_file)).api_key == "from-file"
    monkeypatch.setenv("LARRY_TEST_API_KEY", "from-env")
    assert _pages(tmp_path, api_key_env_var="LARRY_TEST_API_KEY", api_key_file=str(key_file)).api_key == "from-env"
