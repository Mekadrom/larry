from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import requests

from larry.image.config.preprocessing.image_dataset_configs import TePapaCachedDatasetConfig
from larry.image.data.preprocessing.image_dataset_providers import TePapaCachedDatasetProvider

API = "https://data.tepapa.govt.nz/collection"


def _media(media_id: int, title: str = "No Known Copyright Restrictions", allows: bool = True) -> dict[str, Any]:
    return {
        "id": media_id, "type": "ImageObject", "width": 4000, "height": 3000,
        "previewUrl": f"https://media.tepapa.govt.nz/collection/{media_id}/preview",
        "rights": {"title": title, "allowsDownload": allows},
    }


def _object(object_id: int, *media: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "id": str(object_id), "type": "Object", "collection": "Photography", "title": f"Street scene {object_id}",
        "caption": f"Street scene {object_id}, by A. Photographer. Te Papa (A.{object_id})",
        "production": [{"title": "A. Photographer; photographer", "contributor": {"title": "A. Photographer"}}],
        "depicts": [{"title": "Streets"}], "refersTo": [{"title": "Towns"}],
        "isTypeOf": [{"title": "gelatin silver negatives"}],
        "hasRepresentation": list(media), **extra,
    }


class Response:
    def __init__(self, status: int, results: list[dict[str, Any]] | None = None, location: str | None = None) -> None:
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self._results = results or []

    def json(self) -> dict[str, Any]:
        return {"results": self._results}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for url {API}?key=secret", response=self)  # type: ignore[arg-type]


class Session:
    def __init__(self, pages: list[Response]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.headers: dict[str, str] = {}

    def post(self, url: str, **kwargs: Any) -> Response:
        self.calls.append(("POST", url, kwargs))
        return self.pages.pop(0)

    def get(self, url: str, **kwargs: Any) -> Response:
        self.calls.append(("GET", url, kwargs))
        return self.pages.pop(0)


def _provider(tmp_path: Path, pages: list[Response], **overrides: Any) -> tuple[TePapaCachedDatasetProvider, Session]:
    key = tmp_path / "key"
    key.write_text("secret")
    config = TePapaCachedDatasetConfig(**{
        "download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), "api_key_env_var": None,
        "api_key_file": str(key), "min_request_interval": 0.0, "retry_backoff": 0.0, **overrides,
    })
    provider = TePapaCachedDatasetProvider(config)
    session = Session(pages)
    provider.__dict__["session"] = session
    return provider, session


def _rows(provider: TePapaCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_the_scroll_is_followed_until_it_runs_out(tmp_path: Path) -> None:
    provider, session = _provider(tmp_path, [
        Response(303, [_object(1, _media(10))], "/scroll/abc?duration=5"),
        Response(303, [_object(2, _media(20))], "/scroll/def?duration=5"),
        Response(204),
    ])

    assert [r["id"] for r in _rows(provider)] == ["10", "20"]
    (method, url, first), *rest = session.calls
    assert (method, url) == ("POST", f"{API}/search/_scroll")
    assert first["allow_redirects"] is False and first["headers"]["x-api-key"] == "secret"
    assert 'hasRepresentation.rights.title:"CC BY 4.0"' in first["params"]["q"]
    assert [u for _, u, _ in rest] == [f"{API}/scroll/abc?duration=5", f"{API}/scroll/def?duration=5"]


def test_only_downloadable_images_with_an_allowed_license_are_kept(tmp_path: Path) -> None:
    record = _object(
        1, _media(10), _media(11, "CC BY-NC-ND 4.0"), _media(12, "CC BY 4.0"), _media(13, allows=False),
    )
    provider, _ = _provider(tmp_path, [Response(303, [record]), Response(204)])

    rows = _rows(provider)

    assert [(r["id"], r["license"], r["license_raw"]) for r in rows] == [
        ("10", "PD", "No Known Copyright Restrictions"), ("12", "CC-BY", "CC BY 4.0"),
    ]
    first = rows[0]
    assert first["source_url"] == "https://collections.tepapa.govt.nz/object/1"
    assert first["creator"] == "A. Photographer"
    assert first["credit_caption"].startswith("Street scene 1")
    assert first["categories"] == ["Streets", "Towns"]
    assert first["image_url"].endswith("/10/preview")


@pytest.mark.parametrize("record", [
    _object(1, _media(10), collection="TaongaMāori"),
    _object(2, _media(20), collection="PacificCultures"),
    _object(3, _media(30), title="Maori terraces"),
    _object(4, _media(40), caption="Group at the marae, Ōtaki"),
    _object(5, _media(50), refersTo=[{"title": "Māori people"}]),
    _object(6, _media(60), description="A carved waka prow."),
    _object(7, _media(70), title="Photos of Type Tert. fossils"),
    _object(8, _media(80), refersTo=[{"title": "Herbarium specimens"}]),
])
def test_taonga_pacific_collections_maori_subjects_and_specimens_are_excluded(
        tmp_path: Path, record: dict[str, Any],
) -> None:
    provider, _ = _provider(tmp_path, [Response(303, [record, _object(9, _media(90))]), Response(204)])

    assert [r["id"] for r in _rows(provider)] == ["90"]


def test_subject_terms_match_whole_words_only(tmp_path: Path) -> None:
    record = _object(1, _media(10), title="Papakura railway station, Waikanae")
    provider, _ = _provider(tmp_path, [Response(303, [record]), Response(204)])

    assert [r["id"] for r in _rows(provider)] == ["10"]


def test_a_failed_scroll_never_reports_the_key(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, [Response(403)] * 3, max_retries=1)

    with pytest.raises(RuntimeError) as raised:
        provider.load_secondary_data()

    assert "secret" not in str(raised.value)
    assert "status 403" in str(raised.value)


def test_multi_page_objects_contribute_only_their_first_images(tmp_path: Path) -> None:
    diary = _object(1, *[_media(100 + i) for i in range(10)], _media(99, "CC BY-NC-ND 4.0"))
    provider, _ = _provider(tmp_path, [Response(303, [diary]), Response(204)], max_images_per_object=3)

    assert [r["id"] for r in _rows(provider)] == ["100", "101", "102"]


def test_the_image_cap_can_be_lifted(tmp_path: Path) -> None:
    diary = _object(1, *[_media(100 + i) for i in range(6)])
    provider, _ = _provider(tmp_path, [Response(303, [diary]), Response(204)], max_images_per_object=None)

    assert len(_rows(provider)) == 6
