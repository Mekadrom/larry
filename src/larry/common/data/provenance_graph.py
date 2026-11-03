import argparse
import dataclasses
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HFValidationError, RepositoryNotFoundError

type Kind = Literal["model", "dataset"]

PROVENANCE_FILE = "provenance.json"


@dataclasses.dataclass(kw_only=True)
class ProvenanceAncestry:
    datasets: list[str] = dataclasses.field(default_factory=list)
    models: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class ProvenanceGraphConfig:
    root: str
    root_kind: Kind = "model"
    aliases: dict[str, str] = dataclasses.field(default_factory=dict)
    ancestry: dict[str, ProvenanceAncestry] = dataclasses.field(default_factory=dict)
    max_nodes: int = 500


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProvenanceNode:
    name: str
    kind: Kind


@dataclasses.dataclass(kw_only=True)
class ProvenanceRecord:
    node: ProvenanceNode
    found: bool
    produced_on: str | None = None
    git_commit: str | None = None
    parents: list[ProvenanceNode] = dataclasses.field(default_factory=list)
    declared_parents: list[ProvenanceNode] = dataclasses.field(default_factory=list)

    @property
    def all_parents(self) -> list[ProvenanceNode]:
        return [*self.parents, *self.declared_parents]


class ProvenanceGraphCrawler:
    def __init__(self, config: ProvenanceGraphConfig) -> None:
        self.config = config
        self.ancestry = {self.canonical(name): ancestry for name, ancestry in config.ancestry.items()}

    def canonical(self, name: str) -> str:
        name = name.rstrip("/")
        return self.config.aliases.get(name, name)

    @staticmethod
    def is_off_hub(name: str) -> bool:
        return name.startswith(("/", "~", ".")) or "://" in name

    def provenance(self, node: ProvenanceNode) -> dict[str, Any] | None:
        local = Path(node.name).expanduser() / PROVENANCE_FILE
        if local.is_file():
            return json.loads(local.read_text())

        if self.is_off_hub(node.name):
            return None

        try:
            return json.loads(Path(hf_hub_download(node.name, PROVENANCE_FILE, repo_type=node.kind)).read_text())
        except (EntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            return None

    def parents(self, ancestry: Mapping[str, list[str]]) -> list[ProvenanceNode]:
        parents = [ProvenanceNode(name=self.canonical(n), kind="dataset") for n in ancestry.get("datasets", [])]
        parents += [ProvenanceNode(name=self.canonical(n), kind="model") for n in ancestry.get("models", [])]
        return list(dict.fromkeys(parents))

    def record(self, node: ProvenanceNode) -> ProvenanceRecord:
        provenance = self.provenance(node)
        declared = self.ancestry.get(node.name)
        declared_parents = self.parents(dataclasses.asdict(declared)) if declared is not None else []
        if provenance is None:
            return ProvenanceRecord(node=node, found=False, declared_parents=declared_parents)

        parents = self.parents(provenance.get("ancestry") or {})
        return ProvenanceRecord(
            node=node,
            found=True,
            produced_on=provenance.get("produced_on"),
            git_commit=provenance.get("git_commit"),
            parents=parents,
            declared_parents=[parent for parent in declared_parents if parent not in parents],
        )

    def crawl(self) -> dict[ProvenanceNode, ProvenanceRecord]:
        records: dict[ProvenanceNode, ProvenanceRecord] = {}
        queue = deque([ProvenanceNode(name=self.canonical(self.config.root), kind=self.config.root_kind)])
        while queue and len(records) < self.config.max_nodes:
            node = queue.popleft()
            if node not in records:
                records[node] = self.record(node)
                queue.extend(parent for parent in records[node].all_parents if parent not in records)

        return records

    @staticmethod
    def label(record: ProvenanceRecord) -> str:
        name = record.node.name.replace('"', "#quot;")
        return name if record.produced_on is None else f"{name}<br/>{record.produced_on[:10]}"

    def mermaid(self, records: dict[ProvenanceNode, ProvenanceRecord]) -> str:
        ids = {node: f"n{i}" for i, node in enumerate(records)}
        lines = ["flowchart BT"]
        for node, record in records.items():
            label = self.label(record)
            lines.append(f'  {ids[node]}["{label}"]' if node.kind == "dataset" else f'  {ids[node]}(["{label}"])')
            if not record.found:
                lines.append(f"  class {ids[node]} leaf")

        for node, record in records.items():
            lines += [f"  {ids[parent]} --> {ids[node]}" for parent in record.parents if parent in ids]
            lines += [f"  {ids[parent]} -.-> {ids[node]}" for parent in record.declared_parents if parent in ids]

        lines.append("  classDef leaf stroke-dasharray: 4 4")
        return "\n".join(lines)

    def markdown(self) -> str:
        return f"```mermaid\n{self.mermaid(self.crawl())}\n```\n"


def main() -> None:
    argparser = argparse.ArgumentParser(
        prog="provenance-graph",
        description="Crawls provenance.json ancestry from a model or dataset and writes a Mermaid lineage graph.",
    )
    argparser.add_argument("root", help="A huggingface repo id or a local directory holding a provenance.json.")
    argparser.add_argument("--kind", choices=["model", "dataset"], default="model")
    argparser.add_argument("--aliases", type=Path, help="JSON mapping of ancestry names (e.g. local paths) to repo ids.")
    argparser.add_argument(
        "--ancestry",
        type=Path,
        help='JSON mapping of names to declared parents, {"name": {"datasets": [...], "models": [...]}}.',
    )
    argparser.add_argument("--max_nodes", type=int, default=500)
    argparser.add_argument("--output", type=Path, required=True)
    args = argparser.parse_args()

    aliases = json.loads(args.aliases.read_text()) if args.aliases else {}
    declared = json.loads(args.ancestry.read_text()) if args.ancestry else {}
    config = ProvenanceGraphConfig(
        root=args.root,
        root_kind=args.kind,
        aliases=aliases,
        ancestry={name: ProvenanceAncestry(**parents) for name, parents in declared.items()},
        max_nodes=args.max_nodes,
    )
    args.output.write_text(ProvenanceGraphCrawler(config).markdown())
