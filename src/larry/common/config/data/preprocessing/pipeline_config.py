import dataclasses
from datetime import datetime, timezone
from typing import Any, Self

from larry.common.utils import git_utils


@dataclasses.dataclass(kw_only=True)
class ProvenanceAncestry:
    datasets: list[str]
    models: list[str]

    @classmethod
    def new_ancestry(cls, dictionary: dict[str, Any]) -> Self:
        datasets = dictionary.get("datasets", [])
        models = dictionary.get("models", [])
        return cls(datasets=datasets, models=models)


@dataclasses.dataclass(kw_only=True)
class Provenance:
    id_columns: list[str]
    produced_on: str
    git_commit: str
    git_is_dirty: bool | None
    ancestry: ProvenanceAncestry

    @classmethod
    def new_provenance(cls, dictionary: dict[str, Any]) -> Self:
        id_columns = dictionary.get("id_columns", [])
        produced_on = datetime.now(timezone.utc).isoformat(timespec="seconds")

        repo_state = git_utils.git_info()
        if repo_state is not None:
            git_commit = repo_state.git_commit
            git_is_dirty = repo_state.git_dirty
        else:
            git_commit = "not a git repo"
            git_is_dirty = None

        ancestry = ProvenanceAncestry.new_ancestry(dictionary.get("ancestry", {}))

        return cls(
            id_columns=id_columns,
            produced_on=produced_on,
            git_commit=git_commit,
            git_is_dirty=git_is_dirty,
            ancestry=ancestry
        )


@dataclasses.dataclass(kw_only=True)
class PipelineConfig:
    """Config for an entire run of preprocessing."""

    parquet_size_mb: int = 500
    default_batch_size: int = 1000
    seed: int = 42

    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)
