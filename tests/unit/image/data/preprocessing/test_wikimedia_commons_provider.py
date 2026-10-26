import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from larry.image.config.preprocessing.image_dataset_configs import WikimediaCommonsCachedDatasetConfig
from larry.image.data.preprocessing.image_dataset_providers import WikimediaCommonsCachedDatasetProvider

CC0, PUBLIC_DOMAIN, COPYRIGHTED, CC_BY_SA = "Q6938433", "Q19652", "Q50423863", "Q14946043"
US_GOVERNMENT = "Q60671452"


def _item(prop: str, qid: str, qualifiers: dict[str, list[str]] | None = None) -> dict[str, Any]:
    statement: dict[str, Any] = {
        "mainsnak": {"property": prop, "datavalue": {"value": {"entity-type": "item", "id": qid}, "type": "wikibase-entityid"}},
    }
    if qualifiers:
        statement["qualifiers"] = {
            q: [{"datavalue": {"value": {"entity-type": "item", "id": v}}} for v in values] for q, values in qualifiers.items()
        }
    return statement


def _quantity(prop: str, amount: int) -> dict[str, Any]:
    return {"mainsnak": {"property": prop, "datavalue": {"value": {"amount": f"+{amount}"}, "type": "quantity"}}}


def _string(prop: str, value: str) -> dict[str, Any]:
    return {"mainsnak": {"property": prop, "datavalue": {"value": value, "type": "string"}}}


def _entity(
        page_id: int, title: str, *, licenses: tuple[str, ...] = (), statuses: tuple[str, ...] = (),
        mime: str = "image/jpeg", width: int | None = 2000, height: int | None = 1500,
        labels: dict[str, str] | None = None, qualifiers: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    statements: dict[str, list[dict[str, Any]]] = {"P1163": [_string("P1163", mime)]}
    if licenses:
        statements["P275"] = [_item("P275", q) for q in licenses]
    if statuses:
        statements["P6216"] = [_item("P6216", q, qualifiers) for q in statuses]
    if width is not None:
        statements["P2049"] = [_quantity("P2049", width)]
    if height is not None:
        statements["P2048"] = [_quantity("P2048", height)]
    return {
        "type": "mediainfo", "id": f"M{page_id}", "pageid": page_id, "ns": 6, "title": f"File:{title}",
        "labels": {k: {"language": k, "value": v} for k, v in (labels or {}).items()}, "statements": statements,
    }


ENTITIES = [
    _entity(1, "Public domain photo.jpg", statuses=(PUBLIC_DOMAIN,), labels={"en": "A fox", "de": "Ein Fuchs"},
            qualifiers={"P459": [US_GOVERNMENT]}),
    _entity(2, "Copyrighted.jpg", licenses=(CC_BY_SA,), statuses=(COPYRIGHTED,)),
    _entity(3, "Small CC0 (scan).png", licenses=(CC0,), mime="image/png", width=640, height=480),
    _entity(4, "Diagram.svg", licenses=(CC0,), mime="image/svg+xml"),
    _entity(5, "No size.jpg", statuses=(PUBLIC_DOMAIN,), width=None, height=None),
]


def _provider(tmp_path: Path, **overrides: Any) -> WikimediaCommonsCachedDatasetProvider:
    config = WikimediaCommonsCachedDatasetConfig(**{
        "download_extract_cache_dir": str(tmp_path), "cache_dir": str(tmp_path / "hf"), **overrides,
    })
    provider = WikimediaCommonsCachedDatasetProvider(config)
    lines = ["["] + [json.dumps(e) + ("," if i < len(ENTITIES) - 1 else "") for i, e in enumerate(ENTITIES)] + ["]"]
    with gzip.open(provider.mediainfo_path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return provider


def _rows(provider: WikimediaCommonsCachedDatasetProvider) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_public_domain_and_cc0_raster_images_with_sizes_are_kept(tmp_path: Path) -> None:
    rows = _rows(_provider(tmp_path))

    assert [r["id"] for r in rows] == ["M1", "M3"]
    fox = rows[0]
    assert fox["caption_en"] == "A fox" and json.loads(fox["captions"]) == {"en": "A fox", "de": "Ein Fuchs"}
    assert fox["copyright_determination_ids"] == [US_GOVERNMENT]
    assert (fox["width"], fox["height"], fox["title"]) == (2000, 1500, "Public_domain_photo.jpg")


def test_large_files_use_a_standard_thumbnail_and_small_ones_the_original(tmp_path: Path) -> None:
    rows = _rows(_provider(tmp_path))

    digest = hashlib.md5(b"Public_domain_photo.jpg").hexdigest()
    assert rows[0]["image_url"] == (
        f"https://upload.wikimedia.org/wikipedia/commons/thumb/{digest[0]}/{digest[:2]}/"
        "Public_domain_photo.jpg/960px-Public_domain_photo.jpg"
    )
    small = hashlib.md5("Small_CC0_(scan).png".encode()).hexdigest()
    assert rows[1]["image_url"] == (
        f"https://upload.wikimedia.org/wikipedia/commons/{small[0]}/{small[:2]}/Small_CC0_(scan).png"
    )


def test_files_without_dimensions_can_be_kept_as_originals(tmp_path: Path) -> None:
    rows = _rows(_provider(tmp_path, require_dimensions=False))

    assert [r["id"] for r in rows] == ["M1", "M3", "M5"]
    assert "/thumb/" not in rows[2]["image_url"]


def test_non_standard_thumbnail_widths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="standard Wikimedia thumbnail width"):
        WikimediaCommonsCachedDatasetProvider(WikimediaCommonsCachedDatasetConfig(
            download_extract_cache_dir=str(tmp_path), thumbnail_width=1000,
        ))


def test_the_dump_is_pinned_to_its_dated_path(tmp_path: Path) -> None:
    config = WikimediaCommonsCachedDatasetConfig(download_extract_cache_dir=str(tmp_path), dump_date="20261019")

    (_, downloader), = config.downloader_configs()

    assert downloader.download_path == "other/wikibase/commonswiki/20261019/commons-20261019-mediainfo.json.gz"
    assert downloader.output_file_name == "commons-20261019-mediainfo.json.gz"


def test_sampling_is_a_deterministic_subset_of_the_same_size(tmp_path: Path) -> None:
    provider = WikimediaCommonsCachedDatasetProvider(WikimediaCommonsCachedDatasetConfig(
        download_extract_cache_dir=str(tmp_path), sample_fraction=0.25,
    ))
    ids = [f"M{i}" for i in range(20000)]

    first = [i for i in ids if provider.sampled(i)]
    second = [i for i in ids if provider.sampled(i)]

    assert first == second
    assert 4600 < len(first) < 5400
