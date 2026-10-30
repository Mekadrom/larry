from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from larry.image.config.preprocessing.image_dataset_configs import NASAImageLibraryCachedDatasetConfig
from larry.image.data.preprocessing.image_dataset_providers import NASAImageLibraryCachedDatasetProvider

API = "https://images-api.nasa.gov"
ASSETS = "https://images-assets.nasa.gov/image"


def _link(nasa_id: str, size: str, width: int, height: int, rel: str = "alternate", ext: str = "jpg") -> dict[str, Any]:
    return {"href": f"{ASSETS}/{nasa_id}/{nasa_id}~{size}.{ext}", "rel": rel, "render": "image",
            "width": width, "height": height}


def _item(nasa_id: str, **data: Any) -> dict[str, Any]:
    return {
        "data": [{"nasa_id": nasa_id, "media_type": "image", "title": f"Mars {nasa_id}", "center": "JPL",
                  "description": "<p>Dunes on Mars.</p>", "keywords": ["Mars"], **data}],
        "links": [
            _link(nasa_id, "thumb", 160, 120, rel="preview"),
            _link(nasa_id, "small", 640, 480),
            _link(nasa_id, "medium", 1280, 960),
            _link(nasa_id, "orig", 4000, 3000, rel="canonical", ext="tif"),
        ],
    }


class FakeApi:
    def __init__(
            self,
            totals: dict[tuple[int, str | None], int],
            pages: dict[tuple[int, str | None, int], list],
            assets: dict[str, list[str]] | None = None,
    ) -> None:
        self.totals = totals
        self.pages = pages
        self.assets = assets or {}
        self.calls: list[dict[str, Any]] = []
        self.asset_lookups: list[str] = []

    def __call__(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if url.endswith("/collection.json"):
            nasa_id = url.rsplit("/", 2)[1]
            self.asset_lookups.append(nasa_id)
            return self.assets[nasa_id]

        params = dict(params or {})
        self.calls.append(params)
        key = (params["year_start"], params.get("center"))
        if params["page_size"] == 1:
            return {"collection": {"metadata": {"total_hits": self.totals.get(key, 0)}, "items": []}}

        items = self.pages.get((*key, params["page"]), [])
        has_next = (*key, params["page"] + 1) in self.pages
        return {"collection": {"items": items, "links": [{"rel": "next"}] if has_next else []}}


def _provider(tmp_path: Path, api: FakeApi, **overrides: Any) -> NASAImageLibraryCachedDatasetProvider:
    config = NASAImageLibraryCachedDatasetConfig(**{
        "download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), "first_year": 2000,
        "last_year": 2001, "min_request_interval": 0.0, "retry_backoff": 0.0, "max_results_per_query": 300, **overrides,
    })
    provider = NASAImageLibraryCachedDatasetProvider(config)
    provider.get_json = api  # type: ignore[method-assign]
    return provider


def _rows(provider: NASAImageLibraryCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_each_year_is_paged_and_rows_carry_us_government_attribution(tmp_path: Path) -> None:
    api = FakeApi(
        totals={(2000, None): 3, (2001, None): 1},
        pages={(2000, None, 1): [_item("A"), _item("B")], (2000, None, 2): [_item("C")], (2001, None, 1): [_item("D")]},
    )
    rows = _rows(_provider(tmp_path, api))

    assert [r["id"] for r in rows] == ["A", "B", "C", "D"]
    row = rows[0]
    assert (row["license"], row["license_raw"]) == ("PD-USGov", "NASA media usage guidelines")
    assert row["source_url"] == "https://images.nasa.gov/details/A"
    assert row["description"] == "Dunes on Mars."
    assert row["image_url"].endswith("A~medium.jpg")
    assert (row["width"], row["height"]) == (4000, 3000)


def test_years_over_the_search_cap_are_split_by_center(tmp_path: Path) -> None:
    api = FakeApi(
        totals={(2000, None): 500, (2000, "JPL"): 250, (2000, "KSC"): 250},
        pages={(2000, "JPL", 1): [_item("J")], (2000, "KSC", 1): [_item("K")]},
    )
    rows = _rows(_provider(tmp_path, api, last_year=2000, split_centers=["JPL", "KSC"]))

    assert sorted(r["id"] for r in rows) == ["J", "K"]


def test_a_center_allowlist_queries_only_those_centers(tmp_path: Path) -> None:
    api = FakeApi(totals={}, pages={(2000, "JPL", 1): [_item("J")], (2001, "JPL", 1): [_item("J2")]})
    rows = _rows(_provider(tmp_path, api, centers=["JPL"]))

    assert [r["id"] for r in rows] == ["J", "J2"]
    assert all(c.get("center") == "JPL" for c in api.calls)


def test_paging_stops_at_the_search_cap(tmp_path: Path) -> None:
    pages = {(2000, None, p): [_item(f"P{p}")] for p in range(1, 6)}
    api = FakeApi(totals={(2000, None): 5}, pages=pages)
    rows = _rows(_provider(tmp_path, api, last_year=2000, page_size=100, max_results_per_query=300))

    assert [r["id"] for r in rows] == ["P1", "P2", "P3"]


def test_insignia_and_third_party_credits_are_excluded(tmp_path: Path) -> None:
    items = [
        _item("A", title="NASA Meatball Logo"),
        _item("B", keywords=["insignia"]),
        _item("C", secondary_creator="© Getty Images"),
        _item("D", description="Image courtesy of a private collector"),
        _item("E"),
    ]
    rows = _rows(_provider(tmp_path, FakeApi(totals={(2000, None): 5}, pages={(2000, None, 1): items}), last_year=2000))

    assert [r["id"] for r in rows] == ["E"]


def _assets(nasa_id: str, *sizes: str) -> list[str]:
    return [f"http://images-assets.nasa.gov/image/{nasa_id}/{nasa_id}~{size}" for size in sizes]


def test_a_listed_medium_image_is_used_without_looking_up_the_assets(tmp_path: Path) -> None:
    api = FakeApi({}, {})
    provider = _provider(tmp_path, api)

    assert provider.image_url("A", _item("A")["links"]) == f"{ASSETS}/A/A~medium.jpg"
    assert api.asset_lookups == []


def test_unlisted_sizes_are_found_in_the_asset_list(tmp_path: Path) -> None:
    api = FakeApi({}, {}, assets={"A": _assets("A", "orig.tif", "large.jpg", "medium.jpg", "small.jpg", "thumb.jpg")})
    provider = _provider(tmp_path, api)
    small_only = [_link("A", "small", 640, 512), _link("A", "orig", 4000, 3000, rel="canonical", ext="tif")]

    assert provider.image_url("A", small_only) == f"{ASSETS}/A/A~medium.jpg"
    assert api.asset_lookups == ["A"]


def test_the_best_available_size_is_used_when_medium_and_large_do_not_exist(tmp_path: Path) -> None:
    api = FakeApi({}, {}, assets={"A": _assets("A", "orig.jpg", "small.jpg"), "B": _assets("B", "small.jpg")})
    provider = _provider(tmp_path, api)

    assert provider.image_url("A", [_link("A", "small", 640, 640)]).endswith("A~orig.jpg")
    assert provider.image_url("B", [_link("B", "small", 640, 640)]).endswith("B~small.jpg")


def test_asset_lookups_can_be_turned_off(tmp_path: Path) -> None:
    api = FakeApi({}, {})
    provider = _provider(tmp_path, api, resolve_asset_sizes=False)

    assert provider.image_url("A", [_link("A", "small", 640, 640)]).endswith("A~small.jpg")
    assert api.asset_lookups == []


def test_unit_names_round_trip_to_search_parameters() -> None:
    assert NASAImageLibraryCachedDatasetProvider.parse_unit("2011_2011/KSC") == {
        "year_start": 2011, "year_end": 2011, "center": "KSC",
    }
    assert NASAImageLibraryCachedDatasetProvider.parse_unit("1_1949") == {"year_start": 1, "year_end": 1949}
