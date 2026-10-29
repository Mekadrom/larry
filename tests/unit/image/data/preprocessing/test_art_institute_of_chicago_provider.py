import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from larry.image.config.preprocessing.image_dataset_configs import ArtInstituteOfChicagoCachedDatasetConfig
from larry.image.data.preprocessing.image_dataset_providers import ArtInstituteOfChicagoCachedDatasetProvider

IIIF = "https://www.artic.edu/iiif/2"


def _artwork(artwork_id: int, *, public: bool = True, image: str | None = "img", **extra: Any) -> dict[str, Any]:
    return {
        "id": artwork_id,
        "is_public_domain": public,
        "image_id": f"{image}-{artwork_id}" if image else None,
        "alt_image_ids": [f"alt-{artwork_id}"],
        "title": f"Work {artwork_id}",
        "artist_display": "Unknown maker\\nFrance",
        "description": "<p>A <em>blue</em> vase &amp; lid.</p>",
        "thumbnail": {"width": 2000, "height": 1000, "alt_text": "A work made of porcelain."},
        "artwork_type_title": "Vessel",
        "medium_display": "Porcelain",
        "term_titles": ["porcelain"],
        **extra,
    }


def _provider(
        tmp_path: Path, artworks: list[dict[str, Any]], **overrides: Any,
) -> ArtInstituteOfChicagoCachedDatasetProvider:
    config = ArtInstituteOfChicagoCachedDatasetConfig(**{
        "download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), **overrides,
    })
    provider = ArtInstituteOfChicagoCachedDatasetProvider(config)
    with tarfile.open(provider.dump_path, "w:bz2") as archive:
        members = [(f"artic-api-data/json/artworks/{a['id']}.json", a) for a in artworks]
        members.append(("artic-api-data/json/agents/1.json", {"id": 1}))
        for name, payload in members:
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return provider


def _rows(provider: ArtInstituteOfChicagoCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_only_public_domain_artworks_with_an_image_are_kept(tmp_path: Path) -> None:
    rows = _rows(_provider(tmp_path, [_artwork(1), _artwork(2, public=False), _artwork(3, image=None)]))

    assert [r["id"] for r in rows] == ["img-1"]
    row = rows[0]
    assert (row["license"], row["license_raw"]) == ("PD", "is_public_domain")
    assert row["description"] == "A blue vase & lid."
    assert row["source_url"] == "https://www.artic.edu/artworks/1"
    assert row["alt_text"] == "A work made of porcelain."
    assert (row["artwork_type"], row["term_titles"]) == ("Vessel", ["porcelain"])


def test_images_are_requested_at_the_sites_width_unless_already_narrower(tmp_path: Path) -> None:
    narrow = _artwork(2, thumbnail={"width": 600, "height": 900})
    rows = {r["artwork_id"]: r for r in _rows(_provider(tmp_path, [_artwork(1), narrow]))}

    assert rows[1]["image_url"] == f"{IIIF}/img-1/full/843,/0/default.jpg"
    assert rows[2]["image_url"] == f"{IIIF}/img-2/full/full/0/default.jpg"


def test_alternate_images_are_opt_in(tmp_path: Path) -> None:
    rows = _rows(_provider(tmp_path, [_artwork(1)], include_alt_images=True))

    assert [(r["id"], r["is_alt_image"]) for r in rows] == [("img-1", False), ("alt-1", True)]
    assert rows[1]["image_url"] == f"{IIIF}/alt-1/full/843,/0/default.jpg"
