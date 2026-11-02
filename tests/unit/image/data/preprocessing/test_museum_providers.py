from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from larry.common.data.licenses import License
from larry.image.config.preprocessing.image_dataset_configs import (
    RijksmuseumCachedDatasetConfig,
    SMKCachedDatasetConfig,
)
from larry.image.data.preprocessing.image_dataset_providers import (
    RijksmuseumCachedDatasetProvider,
    SMKCachedDatasetProvider,
)

PD = "http://creativecommons.org/publicdomain/mark/1.0/"
OAI = 'xmlns="http://www.openarchives.org/OAI/2.0/"'
RDF = (
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" xmlns:edm="http://www.europeana.eu/schemas/edm/" '
    'xmlns:edmfp="http://www.europeanafashion.eu/edmfp/" xmlns:ore="http://www.openarchives.org/ore/terms/" '
    'xmlns:skos="http://www.w3.org/2004/02/skos/core#" xmlns:svcs="http://rdfs.org/sioc/services#"'
)


def _record(
        object_id: str, rights: str = PD, deleted: bool = False, english: bool = False, service: bool = True,
) -> str:
    if deleted:
        return f'<record><header status="deleted"><identifier>{object_id}</identifier></header></record>'

    image = f"https://iiif.micr.io/{object_id}/full/max/0/default.jpg"
    english_text = '<dc:description xml:lang="en">A knight in armour.</dc:description>' if english else ""
    web = (
        f'<edm:WebResource rdf:about="{image}"><svcs:has_service rdf:resource="https://iiif.micr.io/{object_id}"/>'
        "</edm:WebResource>"
    ) if service else ""
    return f"""<record><header><identifier>https://id.rijksmuseum.nl/{object_id}</identifier></header><metadata>
<rdf:RDF {RDF}>
  <ore:Aggregation rdf:about="a">
    <edm:isShownAt rdf:resource="https://www.rijksmuseum.nl/nl/collectie/object/{object_id}"/>
    <edm:isShownBy rdf:resource="{image}"/>
    <edm:rights rdf:resource="{rights}"/>
  </ore:Aggregation>
  <edm:ProvidedCHO rdf:about="https://id.rijksmuseum.nl/{object_id}">
    <dc:identifier>RP-P-{object_id}</dc:identifier>
    <dc:title xml:lang="nl">Portret van Dirk I</dc:title>
    <dc:description xml:lang="nl">Een ridder in harnas.</dc:description>{english_text}
    <dc:creator rdf:resource="https://id.rijksmuseum.nl/agent"/>
    <dc:type rdf:resource="https://id.rijksmuseum.nl/print"/>
    <dcterms:medium rdf:resource="https://id.rijksmuseum.nl/paper"/>
    <dcterms:created xml:lang="en">1620</dcterms:created>
  </edm:ProvidedCHO>
  <edm:Agent rdf:about="https://id.rijksmuseum.nl/agent">
    <skos:prefLabel xml:lang="nl">Adriaen Matham</skos:prefLabel>
  </edm:Agent>
  <skos:Concept rdf:about="https://id.rijksmuseum.nl/print">
    <skos:prefLabel xml:lang="en">print</skos:prefLabel><skos:prefLabel xml:lang="nl">prent</skos:prefLabel>
  </skos:Concept>
  <skos:Concept rdf:about="https://id.rijksmuseum.nl/paper">
    <skos:prefLabel xml:lang="nl">papier</skos:prefLabel>
  </skos:Concept>
  {web}
</rdf:RDF></metadata></record>"""


def _page(*records: str, token: str | None = None) -> bytes:
    resumption = f"<resumptionToken>{token}</resumptionToken>" if token else "<resumptionToken/>"
    return f"<OAI-PMH {OAI}><ListRecords>{''.join(records)}{resumption}</ListRecords></OAI-PMH>".encode()


IDENTIFY = f"<OAI-PMH {OAI}><Identify><earliestDatestamp>2026-10-13T14:02:15Z</earliestDatestamp></Identify></OAI-PMH>"
NO_RECORDS = f'<OAI-PMH {OAI}><error code="noRecordsMatch">none</error></OAI-PMH>'
OCTOBER = "20261001T000000Z_20261031T235959Z"


def _identifiers(size: int | None = None, headers: int = 0) -> bytes:
    token = f'<resumptionToken completeListSize="{size}">t</resumptionToken>' if size is not None else ""
    body = "<header><identifier>x</identifier></header>" * headers
    return f"<OAI-PMH {OAI}><ListIdentifiers>{body}{token}</ListIdentifiers></OAI-PMH>".encode()


class Response:
    def __init__(self, body: bytes | dict[str, Any]) -> None:
        self.body = body
        self.status_code = 200

    @property
    def content(self) -> bytes:
        assert isinstance(self.body, bytes)
        return self.body

    def json(self) -> Any:
        return self.body

    def raise_for_status(self) -> None:
        ...


class Session:
    def __init__(self, pages: list[bytes | dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> Response:
        self.calls.append(dict(params or {}))
        return Response(self.pages.pop(0))


def _provider[P](
        provider_type: type[P], config_type: Any, tmp_path: Path, pages: list[Any], **overrides: Any,
) -> tuple[P, Session]:
    (tmp_path / "dl").mkdir(parents=True, exist_ok=True)
    config = config_type(**{
        "download_extract_cache_dir": str(tmp_path / "dl"), "cache_dir": str(tmp_path / "hf"),
        "min_request_interval": 0.0, "retry_backoff": 0.0, **overrides,
    })
    provider = provider_type(config)
    session = Session(pages)
    provider.__dict__["session"] = session  # type: ignore[attr-defined]
    return provider, session


def _rows(provider: Any) -> list[dict[str, Any]]:
    provider.load_secondary_data()
    return pq.read_table(provider.metadata_path).to_pylist()


def test_rijksmuseum_harvests_monthly_windows_and_follows_resumption_tokens(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, session = _provider(RijksmuseumCachedDatasetProvider, RijksmuseumCachedDatasetConfig, tmp_path, [
        _page(_record("1"), _record("2", rights="http://rightsstatements.org/vocab/InC/1.0/"), token="t1"),
        _page(_record("3", english=True), _record("4", deleted=True)),
    ])
    monkeypatch.setattr(provider, "units", lambda: iter([OCTOBER]))

    rows = _rows(provider)

    assert session.calls[0] == {
        "verb": "ListRecords", "metadataPrefix": "edm", "from": "2026-10-01T00:00:00Z", "until": "2026-10-31T23:59:59Z",
    }
    assert session.calls[1] == {"verb": "ListRecords", "resumptionToken": "t1"}
    assert [r["object_number"] for r in rows] == ["RP-P-1", "RP-P-3"]
    first, english = rows
    assert first["license"] == License.PUBLIC_DOMAIN
    assert first["image_url"] == "https://iiif.micr.io/1/full/!960,960/0/default.jpg"
    assert (first["title"], first["description"], first["text_language"]) == (
        "Portret van Dirk I", "Een ridder in harnas.", "nl",
    )
    assert first["creator"] == "Adriaen Matham"
    assert first["types"] == ["print"] and first["materials"] == ["papier"] and first["date"] == "1620"
    assert (english["description"], english["text_language"]) == ("A knight in armour.", "en")
    assert english["description_nl"] == "Een ridder in harnas."


def test_rijksmuseum_months_are_halved_until_each_window_is_small_enough(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "larry.image.data.preprocessing.image_dataset_providers.datetime",
        type("Clock", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 11, 5, tzinfo=UTC))}),
    )
    provider, session = _provider(RijksmuseumCachedDatasetProvider, RijksmuseumCachedDatasetConfig, tmp_path, [
        IDENTIFY.encode(),
        _identifiers(size=250),
        _identifiers(size=90),
        _identifiers(headers=60),
        _identifiers(size=160),
        NO_RECORDS.encode(),
        _identifiers(size=40),
    ], max_records_per_unit=100)

    units = list(provider.units())

    assert units == [
        "20261001T000000Z_20261016T115959Z",
        "20261016T120000Z_20261031T235959Z",
        "20261116T000000Z_20261130T235959Z",
    ]
    assert all(c["verb"] in ("Identify", "ListIdentifiers") for c in session.calls)
    assert (session.calls[1]["from"], session.calls[1]["until"]) == ("2026-10-01T00:00:00Z", "2026-10-31T23:59:59Z")
    assert len(session.calls) == 7


def test_rijksmuseum_falls_back_to_the_full_image_without_an_iiif_service(tmp_path: Path) -> None:
    provider, _ = _provider(RijksmuseumCachedDatasetProvider, RijksmuseumCachedDatasetConfig, tmp_path, [
        _page(_record("1", service=False)),
    ])
    provider.__dict__["units"] = lambda: iter([OCTOBER])

    rows = _rows(provider)

    assert rows[0]["image_url"] == "https://iiif.micr.io/1/full/max/0/default.jpg"


def _smk_item(number: str, **extra: Any) -> dict[str, Any]:
    return {
        "object_number": number, "public_domain": True, "rights": "https://creativecommons.org/publicdomain/mark/1.0/",
        "image_iiif_id": f"https://iip.smk.dk/iiif/jp2/{number}.tif.jp2", "image_width": 4992, "image_height": 6287,
        "image_thumbnail": f"https://iip-thumb.smk.dk/iiif/jp2/{number}.tif.jp2/full/!1024,/0/default.jpg",
        "frontend_url": f"https://open.smk.dk/artwork/image/{number}",
        "titles": [{"title": "Augustus og sibyllen", "language": "dansk"}],
        "production": [{"creator": "Trento, Antonio da"}, {"creator": "Parmigianino"}],
        "production_date": [{"period": "1500-1550"}],
        "object_names": [{"name": "Clairobscurtræsnit"}], "techniques": ["Chiaroscuro woodcut"],
        **extra,
    }


def test_smk_pages_by_offset_and_prefers_english_titles(tmp_path: Path) -> None:
    provider, session = _provider(SMKCachedDatasetProvider, SMKCachedDatasetConfig, tmp_path, [
        {"found": 4, "items": [_smk_item("A"), _smk_item("B", public_domain=False)]},
        {"found": 4, "items": [
            _smk_item("C", titles=[
                {"title": "Augustus og sibyllen", "language": "dansk"},
                {"title": "Augustus and the Sibyl", "language": "Engelsk"},
            ], content_description=["Bog med <b>12</b> litografier."]),
            _smk_item("D", image_thumbnail=None),
        ]},
    ], page_size=2)

    rows = _rows(provider)

    assert [c["offset"] for c in session.calls] == [0, 2]
    assert session.calls[0]["filters"] == "[has_image:true],[public_domain:true]"
    assert [r["id"] for r in rows] == ["A", "C"]
    first, english = rows
    assert first["image_url"] == "https://iip-thumb.smk.dk/iiif/jp2/A.tif.jp2/full/!1024,/0/default.jpg"
    assert first["creator"] == "Trento, Antonio da; Parmigianino"
    assert (first["title"], first["text_language"], first["date"]) == ("Augustus og sibyllen", "da", "1500-1550")
    assert (english["title"], english["text_language"]) == ("Augustus and the Sibyl", "en")
    assert english["description"] == "Bog med 12 litografier."
