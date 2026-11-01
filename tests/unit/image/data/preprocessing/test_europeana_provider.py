from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from larry.common.data.licenses import License
from larry.image.config.preprocessing.image_dataset_configs import EuropeanaCachedDatasetConfig
from larry.image.data.preprocessing.image_dataset_providers import EuropeanaCachedDatasetProvider

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
MARK = "http://creativecommons.org/publicdomain/mark/1.0/"


def _item(record_id: str, provider: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"/91617/{record_id}",
        "dataProvider": [provider],
        "provider": ["Swedish Open Cultural Heritage"],
        "country": ["Sweden"],
        "language": ["sv"],
        "rights": [CC0],
        "edmIsShownBy": [f"https://mm.dimu.org/image/{record_id}?dimension=1200x1200"],
        "edmIsShownAt": [f"https://digitaltmuseum.se/{record_id}"],
        "edmDatasetName": ["91617_Ag_SE_Upplandsmuseet"],
        "title": ["Pressure", "Tryck"],
        "dcTitleLangAware": {"sv": ["Tryck"], "def": ["Tryck"], "en": ["Pressure"]},
        "dcDescription": ["Watercolor on paper."],
        "dcDescriptionLangAware": {"sv": ["Akvarell på papper.", "Mönstrad i svart och rött."], "en": ["Watercolor."]},
        "dcCreatorLangAware": {"def": ["A. Konstnär"]},
        "dcSubjectLangAware": {"sv": ["Konstverk"], "def": ["http://data.europeana.eu/concept/2716"]},
        "dcTypeLangAware": {"sv": ["Tryck"]},
        **extra,
    }


def _facets(*providers: str) -> dict[str, Any]:
    return {"facets": [{"name": "DATA_PROVIDER", "fields": [{"label": p, "count": 1} for p in providers]}]}


class Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self.body

    def raise_for_status(self) -> None:
        ...


class Session:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[list[tuple[str, str]]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: list[tuple[str, str]] | None = None, **kwargs: Any) -> Response:
        self.calls.append(list(params or []))
        return Response(self.pages.pop(0))


def _provider(
        tmp_path: Path, pages: list[dict[str, Any]], **overrides: Any,
) -> tuple[EuropeanaCachedDatasetProvider, Session]:
    (tmp_path / "dl").mkdir(parents=True, exist_ok=True)
    key = tmp_path / "key"
    key.write_text("secret")
    config = EuropeanaCachedDatasetConfig(**{
        "download_extract_cache_dir": str(tmp_path / "dl"), "cache_dir": str(tmp_path / "hf"),
        "api_key_env_var": None, "api_key_file": str(key), "min_request_interval": 0.0, "retry_backoff": 0.0,
        **overrides,
    })
    provider = EuropeanaCachedDatasetProvider(config)
    session = Session(pages)
    provider.__dict__["session"] = session
    return provider, session


def _rows(provider: EuropeanaCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_each_selected_provider_is_paged_by_cursor(tmp_path: Path) -> None:
    provider, session = _provider(tmp_path, [
        _facets("Upplands Museum", "Rijksmuseum", 'The "Quoted" Archive'),
        {"items": [_item("a", "Upplands Museum")], "nextCursor": "c1"},
        {"items": [_item("b", "Upplands Museum")], "nextCursor": "c2"},
        {"items": []},
        {"items": [_item("c", 'The "Quoted" Archive')]},
    ])

    rows = _rows(provider)

    assert [r["id"] for r in rows] == ["91617/a", "91617/b", "91617/c"]
    facet, first, second, _, quoted = session.calls
    assert ("facet", "DATA_PROVIDER") in facet
    assert ("qf", 'DATA_PROVIDER:"Upplands Museum"') in first and ("cursor", "*") in first
    assert ("cursor", "c1") in second
    assert ("qf", 'DATA_PROVIDER:"The \\"Quoted\\" Archive"') in quoted
    assert {("qf", f'RIGHTS:"{CC0}"'), ("qf", f'RIGHTS:"{MARK}"'), ("qf", "IMAGE_SIZE:large")} <= set(first)
    assert ("wskey", "secret") in first
    assert not any("Rijksmuseum" in value for call in session.calls[1:] for _, value in call)


def test_only_the_records_own_language_is_kept(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, [_facets("Upplands Museum"), {"items": [_item("a", "Upplands Museum")]}])

    row = _rows(provider)[0]

    assert row["title"] == "Tryck"
    assert row["description"] == "Akvarell på papper. Mönstrad i svart och rött."
    assert row["creator"] == "A. Konstnär"
    assert row["subjects"] == ["Konstverk"]
    assert row["text_language"] == "sv"
    assert row["license"] == License.CC0 and row["license_raw"] == CC0
    assert row["source_url"] == "https://digitaltmuseum.se/a"
    assert row["image_url"] == "https://mm.dimu.org/image/a?dimension=1200x1200"


def test_a_translation_alone_is_not_used(tmp_path: Path) -> None:
    item = _item("a", "Upplands Museum", dcTitleLangAware={"en": ["Pressure"]}, dcDescriptionLangAware={"en": ["x"]})
    provider, _ = _provider(tmp_path, [_facets("Upplands Museum"), {"items": [item]}])

    row = _rows(provider)[0]

    assert row["title"] is None and row["description"] is None


def test_specimen_providers_are_flagged_and_unusable_records_skipped(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, [
        _facets("Naturalis Biodiversity Center"),
        {"items": [
            _item("a", "Naturalis Biodiversity Center"),
            _item("b", "Naturalis Biodiversity Center", rights=["http://creativecommons.org/licenses/by/4.0/"]),
            _item("c", "Naturalis Biodiversity Center", edmIsShownBy=[]),
            _item("e", "Naturalis Biodiversity Center", edmIsShownBy=None, edmPreview=[
                "https://api.europeana.eu/thumbnail/v2/url.json?uri=https%3A%2F%2Fservice.archief.nl%2Fiip%2Fx.jp2"
                "%2Ffull%2Fmax%2F0%2Fdefault.jpg&type=IMAGE"
            ]),
            _item("d", "Naturalis Biodiversity Center", rights=[MARK]),
        ]},
    ])

    rows = _rows(provider)

    assert [r["id"] for r in rows] == ["91617/a", "91617/e", "91617/d"]
    assert all(r["generation_excluded"] and r["generation_exclusion_reasons"] == ["specimen"] for r in rows)
    assert rows[1]["image_url"] == "https://service.archief.nl/iip/x.jp2/full/max/0/default.jpg"
    assert rows[2]["license"] == License.PUBLIC_DOMAIN


def test_a_split_reuses_the_shared_shards(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    main, _ = _provider(tmp_path / "main", [
        _facets("Upplands Museum", "Naturalis Biodiversity Center"),
        {"items": [_item("a", "Upplands Museum")]},
        {"items": [_item("b", "Naturalis Biodiversity Center")]},
    ], shards_dir=str(shards), excluded_providers=[])
    main.load_secondary_data()

    split, session = _provider(tmp_path / "split", [_facets("Upplands Museum", "Naturalis Biodiversity Center")],
                               shards_dir=str(shards), providers=["Naturalis Biodiversity Center"])

    assert [r["id"] for r in _rows(split)] == ["91617/b"]
    assert len(session.calls) == 1


def test_a_truncated_provider_list_is_refused(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, [_facets("A", "B")], max_provider_facets=2)

    with pytest.raises(ValueError, match="max_provider_facets"):
        list(provider.units())
