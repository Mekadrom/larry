import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import larry.common.data.provenance_graph as provenance_graph
from larry.common.data.provenance_graph import (
    ProvenanceAncestry,
    ProvenanceGraphConfig,
    ProvenanceGraphCrawler,
    ProvenanceNode,
)


def _provenance(
        datasets: list[str] | None = None, models: list[str] | None = None, date: str = "2026-10-25T23:45:58+00:00",
) -> dict[str, Any]:
    ancestry = {"datasets": datasets or [], "models": models or []}
    return {"ancestry": ancestry, "produced_on": date, "git_commit": "abc123"}


def _local(root: Path, name: str, provenance: dict[str, Any]) -> str:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "provenance.json").write_text(json.dumps(provenance))
    return str(directory)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[tuple[str, str], dict[str, Any] | None]:
    repos: dict[tuple[str, str], dict[str, Any] | None] = {}
    calls: list[tuple[str, str]] = []

    def download(repo_id: str, filename: str, repo_type: str) -> str:
        calls.append((repo_id, repo_type))
        if (repo_id, repo_type) not in repos:
            request = httpx.Request("GET", f"https://huggingface.co/{repo_id}")
            raise RepositoryNotFoundError(repo_id, response=httpx.Response(404, request=request))

        provenance = repos[(repo_id, repo_type)]
        if provenance is None:
            raise EntryNotFoundError(filename)

        path = tmp_path / "hub" / repo_type / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance))
        return str(path)

    monkeypatch.setattr(provenance_graph, "hf_hub_download", download)
    repos[("__calls__", "")] = {"calls": calls}
    return repos


def _crawler(root: str, kind: str = "model", **overrides: Any) -> ProvenanceGraphCrawler:
    return ProvenanceGraphCrawler(ProvenanceGraphConfig(root=root, root_kind=kind, **overrides))


def test_ancestry_is_followed_through_models_and_datasets_to_external_leaves(hub: dict[Any, Any]) -> None:
    hub[("org/encoder", "model")] = _provenance(datasets=["org/index", "org/normalized"])
    hub[("org/index", "dataset")] = _provenance(datasets=["org/normalized"], models=["facebook/wav2vec2"])
    hub[("org/normalized", "dataset")] = _provenance(datasets=["upstream/corpus"])
    hub[("upstream/corpus", "dataset")] = None

    records = _crawler("org/encoder").crawl()

    found = {node.name: record.found for node, record in records.items()}
    assert found == {
        "org/encoder": True, "org/index": True, "org/normalized": True,
        "facebook/wav2vec2": False, "upstream/corpus": False,
    }
    index = records[ProvenanceNode(name="org/index", kind="dataset")]
    assert index.parents == [ProvenanceNode(name="org/normalized", kind="dataset"),
                             ProvenanceNode(name="facebook/wav2vec2", kind="model")]
    assert index.produced_on == "2026-10-25T23:45:58+00:00" and index.git_commit == "abc123"
    calls = hub[("__calls__", "")]["calls"]
    assert calls.count(("org/normalized", "dataset")) == 1
    assert ("facebook/wav2vec2", "model") in calls


def test_local_directories_and_aliases_resolve_before_the_hub(tmp_path: Path, hub: dict[Any, Any]) -> None:
    aligned = _local(tmp_path, "scotus-aligned", _provenance(datasets=["org/scotus-unaligned"]))
    hub[("org/scotus-index", "dataset")] = _provenance(datasets=["~/.cache/larry/datasets/scotus-aligned/"])
    hub[("org/scotus-unaligned", "dataset")] = _provenance()

    records = _crawler(
        "org/scotus-index", kind="dataset", aliases={"~/.cache/larry/datasets/scotus-aligned": aligned},
    ).crawl()

    assert [node.name for node in records] == ["org/scotus-index", aligned, "org/scotus-unaligned"]
    assert all(record.found for record in records.values())


@pytest.mark.parametrize("parent", ["/nowhere/normalized", "https://datashare.ed.ac.uk/handle/10283/3443"])
def test_local_paths_and_urls_are_leaves_without_a_hub_lookup(hub: dict[Any, Any], parent: str) -> None:
    hub[("org/index", "dataset")] = _provenance(datasets=[parent])

    records = _crawler("org/index", kind="dataset").crawl()

    assert not records[ProvenanceNode(name=parent, kind="dataset")].found
    assert (parent, "dataset") not in hub[("__calls__", "")]["calls"]


def test_names_that_are_not_repo_ids_are_leaves(hub: dict[Any, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    hub[("org/index", "dataset")] = _provenance(datasets=["not a repo id"])
    download = provenance_graph.hf_hub_download

    def validating(repo_id: str, filename: str, repo_type: str) -> str:
        if " " in repo_id:
            raise provenance_graph.HFValidationError(repo_id)

        return download(repo_id, filename, repo_type)

    monkeypatch.setattr(provenance_graph, "hf_hub_download", validating)

    assert not _crawler("org/index", kind="dataset").crawl()[ProvenanceNode(name="not a repo id", kind="dataset")].found


def test_cycles_and_the_node_cap_end_the_crawl(hub: dict[Any, Any]) -> None:
    hub[("org/a", "dataset")] = _provenance(datasets=["org/b"])
    hub[("org/b", "dataset")] = _provenance(datasets=["org/a", "org/c"])
    hub[("org/c", "dataset")] = _provenance()

    assert len(_crawler("org/a", kind="dataset").crawl()) == 3
    assert len(_crawler("org/a", kind="dataset", max_nodes=2).crawl()) == 2


def test_the_mermaid_graph_draws_kinds_leaves_and_edges(hub: dict[Any, Any]) -> None:
    hub[("org/encoder", "model")] = _provenance(datasets=["org/index"])
    hub[("org/index", "dataset")] = _provenance(datasets=["upstream/corpus"], date="2026-10-19T00:00:00+00:00")

    crawler = _crawler("org/encoder")
    text = crawler.mermaid(crawler.crawl())

    assert text.splitlines() == [
        "flowchart BT",
        '  n0(["org/encoder<br/>2026-10-25"])',
        '  n1["org/index<br/>2026-10-19"]',
        '  n2["upstream/corpus"]',
        "  class n2 leaf",
        "  n1 --> n0",
        "  n2 --> n1",
        "  classDef leaf stroke-dasharray: 4 4",
    ]


def test_the_cli_writes_a_mermaid_markdown_block(
        tmp_path: Path, hub: dict[Any, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub[("org/encoder", "model")] = _provenance()
    output = tmp_path / "lineage.md"
    monkeypatch.setattr("sys.argv", ["provenance-graph", "org/encoder", "--output", str(output)])

    provenance_graph.main()

    text = output.read_text()
    assert text.startswith("```mermaid\nflowchart BT\n") and text.endswith("```\n")


def test_declared_ancestry_extends_nodes_without_provenance(hub: dict[Any, Any]) -> None:
    hub[("org/index", "dataset")] = _provenance(datasets=["mythicinfinity/libriheavy"])
    hub[("mythicinfinity/libriheavy", "dataset")] = None
    ancestry = {
        "mythicinfinity/libriheavy": ProvenanceAncestry(models=["Zengwei/zipformer"]),
        "Zengwei/zipformer": ProvenanceAncestry(datasets=["openslr/librispeech_asr"]),
    }

    records = _crawler("org/index", kind="dataset", ancestry=ancestry).crawl()

    libriheavy = records[ProvenanceNode(name="mythicinfinity/libriheavy", kind="dataset")]
    zipformer = records[ProvenanceNode(name="Zengwei/zipformer", kind="model")]
    assert not libriheavy.found and libriheavy.declared_parents == [zipformer.node]
    assert zipformer.declared_parents == [ProvenanceNode(name="openslr/librispeech_asr", kind="dataset")]
    assert ProvenanceNode(name="openslr/librispeech_asr", kind="dataset") in records


def test_declared_ancestry_adds_to_recorded_ancestry_without_duplicating_it(hub: dict[Any, Any]) -> None:
    hub[("org/index", "dataset")] = _provenance(datasets=["org/normalized"])
    hub[("org/normalized", "dataset")] = _provenance()
    ancestry = {"org/index": ProvenanceAncestry(datasets=["org/normalized"], models=["org/aligner"])}

    record = _crawler("org/index", kind="dataset", ancestry=ancestry).crawl()[
        ProvenanceNode(name="org/index", kind="dataset")
    ]

    assert record.parents == [ProvenanceNode(name="org/normalized", kind="dataset")]
    assert record.declared_parents == [ProvenanceNode(name="org/aligner", kind="model")]


def test_declared_ancestry_keys_and_parents_go_through_aliases(hub: dict[Any, Any]) -> None:
    hub[("org/index", "dataset")] = _provenance(datasets=["~/local/libriheavy"])
    ancestry = {"~/local/libriheavy": ProvenanceAncestry(models=["~/local/aligner"])}
    aliases = {"~/local/libriheavy": "mythicinfinity/libriheavy", "~/local/aligner": "Zengwei/zipformer"}

    records = _crawler("org/index", kind="dataset", aliases=aliases, ancestry=ancestry).crawl()

    assert records[ProvenanceNode(name="mythicinfinity/libriheavy", kind="dataset")].declared_parents == [
        ProvenanceNode(name="Zengwei/zipformer", kind="model"),
    ]


def test_declared_edges_are_dotted(hub: dict[Any, Any]) -> None:
    hub[("org/index", "dataset")] = _provenance(datasets=["upstream/corpus"])
    ancestry = {"upstream/corpus": ProvenanceAncestry(models=["org/aligner"])}

    crawler = _crawler("org/index", kind="dataset", ancestry=ancestry)
    lines = crawler.mermaid(crawler.crawl()).splitlines()

    assert "  n1 --> n0" in lines and "  n2 -.-> n1" in lines


def test_the_cli_reads_declared_ancestry(tmp_path: Path, hub: dict[Any, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    hub[("org/encoder", "model")] = _provenance()
    ancestry = tmp_path / "ancestry.json"
    ancestry.write_text(json.dumps({"org/encoder": {"datasets": ["upstream/corpus"]}}))
    output = tmp_path / "lineage.md"
    monkeypatch.setattr(
        "sys.argv", ["provenance-graph", "org/encoder", "--ancestry", str(ancestry), "--output", str(output)],
    )

    provenance_graph.main()

    assert "-.->" in output.read_text() and "upstream/corpus" in output.read_text()
