import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from larry.image.config.preprocessing.image_dataset_configs import SmithsonianOpenAccessCachedDatasetConfig
from larry.image.data.preprocessing.image_dataset_providers import SmithsonianOpenAccessCachedDatasetProvider

BUCKET = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com"
IDS = "https://ids.si.edu/ids/deliveryService"


def _media(ids_id: str, *, kind: str = "Images", access: str = "CC0", size: tuple[int, int] | None = (3000, 2000),
           resource_size: str | None = None) -> dict[str, Any]:
    media: dict[str, Any] = {"idsId": ids_id, "guid": f"guid-{ids_id}", "type": kind, "usage": {"access": access}}
    if size is not None:
        media["width"], media["height"] = size
    media["resources"] = [{"label": "High-resolution JPEG", "dimensions": resource_size}] if resource_size else []
    return media


def _record(record_id: str, unit: str, *media: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"edan-{record_id}",
        "unitCode": unit.upper(),
        "content": {
            "freetext": {
                "notes": [
                    {"label": "Description", "content": f"About {record_id}"},
                    {"label": "Location", "content": "Currently not on view"},
                    {"label": "Summary", "content": "Summary"},
                ],
                "name": [{"label": "maker", "content": "Jane Doe"}],
            },
            "indexedStructured": {"object_type": ["fire mark"]},
            "descriptiveNonRepeating": {
                "record_ID": record_id,
                "guid": f"ark-{record_id}",
                "unit_code": unit.upper(),
                "data_source": f"Museum {unit}",
                "title": {"label": "Title", "content": f"Title {record_id}"},
                "online_media": {"media": list(media)},
            },
        },
    }


FILES = {
    "nmah/00.txt": [
        _record("nmah_1", "nmah", _media("NMAH-1"), _media("NMAH-2", size=(800, 600))),
        _record("nmah_2", "nmah", _media("NMAH-3", access="Usage conditions apply")),
        _record("nmah_3", "nmah", _media("NMAH-4", kind="Scanned books")),
        _record("nmah_4", "nmah"),
    ],
    "nmah/01.txt": [
        _record("nmah_5", "nmah", _media("NMAH-1")),
    ],
    "nmnhbotany/00.txt": [
        _record("bot_1", "nmnhbotany", _media("ark:/65665/m3abc", size=None, resource_size="3200x2132")),
        _record("bot_2", "nmnhbotany", _media("ark:/65665/m3def", size=None)),
    ],
}


def _provider(tmp_path: Path, **overrides: Any) -> SmithsonianOpenAccessCachedDatasetProvider:
    config = SmithsonianOpenAccessCachedDatasetConfig(**{
        "download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), **overrides,
    })
    provider = SmithsonianOpenAccessCachedDatasetProvider(config)
    root = tmp_path / config.metadata_prefix
    units = sorted({path.split("/")[0] for path in FILES})
    root.mkdir(parents=True)
    (root / "index.txt").write_text("".join(f"{BUCKET}/{config.metadata_prefix}/{u}/index.txt\n" for u in units))
    for unit in units:
        (root / unit).mkdir()
        paths = [p for p in FILES if p.startswith(f"{unit}/")]
        (root / unit / "index.txt").write_text("".join(f"{BUCKET}/{config.metadata_prefix}/{p}\n" for p in paths))
    for path, records in FILES.items():
        (root / path).write_text("".join(json.dumps(r) + "\n" for r in records))
    return provider


def _rows(provider: SmithsonianOpenAccessCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_only_cc0_images_are_kept_once_each(tmp_path: Path) -> None:
    rows = _rows(_provider(tmp_path))

    assert [r["id"] for r in rows] == ["NMAH-1", "NMAH-2", "ark:/65665/m3abc", "ark:/65665/m3def"]
    first = rows[0]
    assert (first["record_id"], first["unit_code"], first["title"]) == ("nmah_1", "NMAH", "Title nmah_1")
    assert first["object_types"] == ["fire mark"]
    assert json.loads(first["freetext"])["notes"][0]["content"] == "About nmah_1"


def test_attribution_columns_are_filled(tmp_path: Path) -> None:
    first = _rows(_provider(tmp_path))[0]

    assert (first["license"], first["license_raw"]) == ("CC0", "CC0")
    assert first["creator"] == "maker: Jane Doe"
    assert first["source_url"] == "ark-nmah_1"
    assert first["description"] == "About nmah_1\n\nSummary"


def test_large_images_are_requested_resized_and_small_ones_whole(tmp_path: Path) -> None:
    rows = {r["id"]: r for r in _rows(_provider(tmp_path))}

    assert rows["NMAH-1"]["image_url"] == f"{IDS}?id=NMAH-1&max=960"
    assert rows["NMAH-2"]["image_url"] == f"{IDS}?id=NMAH-2"
    assert rows["ark:/65665/m3abc"]["image_url"] == f"{IDS}?id=ark:/65665/m3abc&max=960"


def test_missing_media_dimensions_fall_back_to_resources_then_to_resizing(tmp_path: Path) -> None:
    rows = {r["id"]: r for r in _rows(_provider(tmp_path))}

    assert (rows["ark:/65665/m3abc"]["width"], rows["ark:/65665/m3abc"]["height"]) == (3200, 2132)
    assert rows["ark:/65665/m3def"]["width"] is None
    assert rows["ark:/65665/m3def"]["image_url"].endswith("&max=960")


def test_units_can_be_selected_and_excluded(tmp_path: Path) -> None:
    assert {r["unit_code"] for r in _rows(_provider(tmp_path / "a", units=["nmah"]))} == {"NMAH"}
    assert {r["unit_code"] for r in _rows(_provider(tmp_path / "b", excluded_units=["nmah"]))} == {"NMNHBOTANY"}


def test_raw_metadata_is_dropped_after_filtering_and_shards_resume(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    rows = _rows(provider)
    raw = tmp_path / provider.config.metadata_prefix / "nmah/00.txt"

    assert not raw.exists()
    provider.metadata_path.unlink()
    assert _rows(provider) == rows


def test_raw_metadata_can_be_kept(tmp_path: Path) -> None:
    provider = _provider(tmp_path, keep_raw_metadata=True)
    _rows(provider)

    assert (tmp_path / provider.config.metadata_prefix / "nmah/00.txt").exists()
