from pathlib import Path
from typing import Any

import pytest
import requests

import larry.common.data.preprocessing.url_mappers as url_mappers
from larry.common.config.data.preprocessing.mapper_configs import UrlBytesMapperConfig
from larry.common.data.preprocessing.url_mappers import UrlBytesMapper
from tests.helpers.common import provenance

TOKEN = "s3cret-token"


@pytest.fixture(autouse=True)
def fresh_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_mappers, "_BEARER_TOKENS", {})
    monkeypatch.delenv("WIKIMEDIA_ACCESS_TOKEN", raising=False)


def _mapper(tmp_path: Path, token: str | None = TOKEN, **overrides: Any) -> UrlBytesMapper:
    path = tmp_path / "token"
    if token is not None:
        path.write_text(f"{token}\n")

    config = UrlBytesMapperConfig(**{
        "download_cache_dir": str(tmp_path / "cache"), "busy_wait": 0.0, "max_retries": 0,
        "bearer_token_env_var": "WIKIMEDIA_ACCESS_TOKEN", "bearer_token_file": str(path),
        "bearer_token_hosts": ["wikimedia.org"], **overrides,
    })
    return UrlBytesMapper(provenance(), config)


def _response(status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = b"payload"
    return response


def _capture(mapper: UrlBytesMapper, monkeypatch: pytest.MonkeyPatch, status: int = 200) -> list[dict[str, str]]:
    sent: list[dict[str, str]] = []
    session = requests.Session()

    def get(url: str, timeout: float, headers: dict[str, str]) -> requests.Response:
        sent.append(headers)
        return _response(status)

    monkeypatch.setattr(session, "get", get)
    monkeypatch.setattr(mapper, "_lazy_get_session", lambda: session)
    return sent


def test_the_token_goes_only_to_configured_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapper = _mapper(tmp_path)
    sent = _capture(mapper, monkeypatch)

    for url in ("https://upload.wikimedia.org/a.jpg", "https://wikimedia.org/b.jpg",
                "https://example.org/c.jpg", "https://notwikimedia.org/d.jpg"):
        mapper.preprocess_example(url)

    assert sent == [{"Authorization": f"Bearer {TOKEN}"}] * 2 + [{}, {}]


def test_the_environment_variable_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKIMEDIA_ACCESS_TOKEN", "from-env")

    assert _mapper(tmp_path).request_headers("https://upload.wikimedia.org/a.jpg") == {
        "Authorization": "Bearer from-env",
    }


def test_no_token_configuration_sends_nothing(tmp_path: Path) -> None:
    mapper = _mapper(tmp_path, bearer_token_env_var=None, bearer_token_file=None, bearer_token_hosts=[])

    assert mapper.request_headers("https://upload.wikimedia.org/a.jpg") == {}


def test_a_configured_but_missing_token_fails_loudly(tmp_path: Path) -> None:
    mapper = _mapper(tmp_path, token=None)

    with pytest.raises(ValueError, match="no bearer token"):
        mapper.request_headers("https://upload.wikimedia.org/a.jpg")

    assert mapper.request_headers("https://example.org/a.jpg") == {}


def test_a_token_without_hosts_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bearer_token_hosts"):
        _mapper(tmp_path, bearer_token_hosts=[])


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_token_stops_preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    mapper = _mapper(tmp_path)
    _capture(mapper, monkeypatch, status)

    with pytest.raises(ValueError, match="bearer token") as error:
        mapper.preprocess_example("https://upload.wikimedia.org/a.jpg")

    assert TOKEN not in str(error.value)


def test_requests_drops_the_header_on_a_redirect_to_another_host() -> None:
    session = requests.Session()

    assert session.should_strip_auth("https://upload.wikimedia.org/a.jpg", "https://example.org/a.jpg")
    assert not session.should_strip_auth("https://upload.wikimedia.org/a.jpg", "https://upload.wikimedia.org/b.jpg")


def test_the_token_stays_out_of_the_mappers_state(tmp_path: Path) -> None:
    mapper = _mapper(tmp_path)
    mapper.request_headers("https://upload.wikimedia.org/a.jpg")

    assert TOKEN not in repr(vars(mapper))
    assert TOKEN not in repr(mapper.config)
