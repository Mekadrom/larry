from pathlib import Path
from typing import Any

import pytest
import requests
from datadiligence.rules import SpawningAPI
from datasets import Dataset

from larry.common.config.data.preprocessing.mapper_configs import UrlBytesMapperConfig
from larry.common.config.data.preprocessing.pruner_configs import DoNotTrainPrunerConfig
from larry.common.data.preprocessing.pruners import DoNotTrainPruner
from larry.common.data.preprocessing.url_mappers import UrlBytesMapper
from tests.helpers.common import provenance

OPTED_OUT = {"https://a.example/2.jpg", "https://b.example/9.jpg"}


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def filter_allowed(self: SpawningAPI, urls: list[str], **kwargs: Any) -> list[str]:
        if self.api_key != "key" or len(set(urls)) != len(urls) or None in urls:
            raise AssertionError(f"unexpected submission: key={self.api_key!r} urls={urls}")
        return [url for url in urls if url not in OPTED_OUT]

    monkeypatch.setenv("SPAWNING_OPTS_KEY", "key")
    monkeypatch.setattr(SpawningAPI, "filter_allowed", filter_allowed)


def _dataset() -> Dataset:
    return Dataset.from_dict({
        "id": ["1", "2", "3", "4"],
        "url": ["https://a.example/1.jpg", "https://a.example/2.jpg", None, "https://a.example/4.jpg"],
        "alt": ["https://b.example/1.jpg", "https://b.example/2.jpg", "https://b.example/3.jpg", "https://b.example/9.jpg"],
    })


def test_opted_out_and_missing_urls_are_pruned(registry: None) -> None:
    pruner = DoNotTrainPruner(provenance(), DoNotTrainPrunerConfig(num_proc=1))

    kept = pruner.preprocess_dataset(_dataset())

    assert kept["id"] == ["1", "4"]


def test_every_url_column_must_be_allowed(registry: None) -> None:
    pruner = DoNotTrainPruner(provenance(), DoNotTrainPrunerConfig(input_columns=["url", "alt"], prune_nulls=False))

    kept = pruner.preprocess_dataset(_dataset())

    assert kept["id"] == ["1", "3"]


def test_a_missing_api_key_fails_instead_of_allowing_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPAWNING_OPTS_KEY", raising=False)

    pruner = DoNotTrainPruner(provenance(), DoNotTrainPrunerConfig())

    with pytest.raises(ValueError, match="SPAWNING_OPTS_KEY"):
        pruner.preprocess_dataset(_dataset())


def _response(headers: dict[str, str]) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = b"payload"
    response.headers.update(headers)
    return response


@pytest.mark.parametrize(("headers", "kept"), [
    ({}, True),
    ({"X-Robots-Tag": "noindex"}, True),
    ({"X-Robots-Tag": "noai"}, False),
    ({"X-Robots-Tag": "noindex, noimageai"}, False),
    ({"tdm-reservation": "1"}, False),
    ({"tdm-reservation": "0"}, True),
])
def test_downloads_that_opt_out_in_their_headers_are_dropped(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], kept: bool,
) -> None:
    config = UrlBytesMapperConfig(download_cache_dir=str(tmp_path / "cache"), busy_wait=0.0)
    mapper = UrlBytesMapper(provenance(), config)
    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda url, timeout: _response(headers))
    monkeypatch.setattr(mapper, "_lazy_get_session", lambda: session)

    result = mapper.preprocess_example("https://a.example/1.jpg")

    assert (result is not None) == kept
    assert any((tmp_path / "cache").iterdir()) == kept


def test_header_checks_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = UrlBytesMapperConfig(
        download_cache_dir=str(tmp_path / "cache"), busy_wait=0.0, respect_opt_out_headers=False
    )
    mapper = UrlBytesMapper(provenance(), config)
    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda url, timeout: _response({"X-Robots-Tag": "noai"}))
    monkeypatch.setattr(mapper, "_lazy_get_session", lambda: session)

    assert mapper.preprocess_example("https://a.example/1.jpg") is not None


def test_the_byte_rate_limit_spaces_downloads_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import larry.common.data.preprocessing.url_mappers as url_mappers

    clock = {"now": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(url_mappers.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(url_mappers.time, "sleep", sleep)
    config = UrlBytesMapperConfig(download_cache_dir=str(tmp_path / "cache"), max_bytes_per_second=1000.0)
    mapper = UrlBytesMapper(provenance(), config)

    for size in (600, 600, 600):
        mapper.throttle(size)

    assert slept == pytest.approx([0.2, 0.6])


def test_no_rate_limit_means_no_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import larry.common.data.preprocessing.url_mappers as url_mappers

    monkeypatch.setattr(url_mappers.time, "sleep", lambda seconds: pytest.fail("slept without a limit"))
    mapper = UrlBytesMapper(provenance(), UrlBytesMapperConfig(download_cache_dir=str(tmp_path / "cache")))

    mapper.throttle(10 ** 9)
