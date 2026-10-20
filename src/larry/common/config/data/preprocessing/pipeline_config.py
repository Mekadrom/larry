import copy
import dataclasses
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from larry.common.config.config import Config, NamedEntries
from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.common.utils import git_utils


@dataclasses.dataclass(kw_only=True)
class ProvenanceAncestry(Config):
    """Helper dataclass for storing provenance ancestry - immediate source datasets and models.

    Shared by preprocessing and training pipelines.
    """

    datasets: set[str] = dataclasses.field(default_factory=set)
    models: set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass(kw_only=True)
class Provenance(Config):
    """Helper dataclass for tracking the provenance of outputs of either preprocessing or training pipelines."""

    id_columns: list[str] = dataclasses.field(default_factory=list)
    ancestry: ProvenanceAncestry = dataclasses.field(default_factory=ProvenanceAncestry)
    produced_on: str | None = None
    git_commit: str | None = None
    git_is_dirty: bool | None = None
    cli: str | None = None
    config: dict[str, Any] | None = None

    def populate_mutables(self) -> None:
        """Populates instantaneous information onto ``self``."""
        self.produced_on = datetime.now(UTC).isoformat(timespec="seconds")

        repo_state = git_utils.git_info()
        if repo_state is not None:
            self.git_commit = repo_state.git_commit
            self.git_is_dirty = repo_state.git_dirty
        else:
            self.git_commit = "not a git repo"
            self.git_is_dirty = None


@dataclasses.dataclass(kw_only=True)
class PipelineConfig(Config):
    """Superclass for configuring the entire pipeline."""

    parquet_size_mb: int = 500
    default_batch_size: int = 1000
    seed: int = 42

    provenance: Provenance = dataclasses.field(default_factory=Provenance)


@dataclasses.dataclass(kw_only=True)
class PipelineDefinition(Config):
    preprocessor: PipelineConfig = dataclasses.field(default_factory=PipelineConfig)
    dataset: DatasetConfig
    splits: dict[str, float] | None = None
    pipeline: list[PreprocessorConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.splits and not math.isclose(sum(self.splits.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"split fractions must sum to 1, got {self.splits}")

    @classmethod
    def __pre_deserialize__(cls, d: dict[Any, Any]) -> dict[Any, Any]:
        return NamedEntries.to_lists(d, ("pipeline",))

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        d = {**d, "dataset": self.dataset.to_dict(), "pipeline": [step.to_dict() for step in self.pipeline]}
        return NamedEntries.to_maps(d, ("pipeline",))


class PipelineState:
    """Delegate for pipeline and preprocessing config parsing functionality and state storage for a run."""

    config: PipelineConfig
    dataset_provider: DatasetProvider
    pipeline: list[Preprocessor]
    provenance: Provenance

    def __init__(self, definition: PipelineDefinition) -> None:
        self.log = logging.getLogger(__name__)

        self.definition = definition
        self.config = definition.preprocessor
        self.fractions = definition.splits

        dataset_config = definition.dataset
        dataset_provider_type = DatasetProvider.REGISTRY.for_config(dataset_config)
        self.log.info(f"Using {dataset_provider_type.__name__} for dataset_config={dataset_config}")
        self.dataset_provider = dataset_provider_type(dataset_config)

        self.provenance = self._build_provenance(copy.deepcopy(self.config.provenance))

        if not definition.pipeline:
            self.log.critical(
                "No pipeline configured; parquets will be downloaded and cached locally according to the configuration."
            )
        self.pipeline = [self._build_step(step) for step in definition.pipeline]
        self.log.info("Preprocessor constructed")

    def _build_step(self, step: PreprocessorConfig) -> Preprocessor:
        preprocessor_type = Preprocessor.REGISTRY.for_config(step)
        self.log.info(f"step {step.name!r}: {preprocessor_type.__name__} using {step}")
        return preprocessor_type(self.provenance, step)

    def _build_provenance(self, provenance: Provenance) -> Provenance:
        """Builds provenance using intuited defaults for dataset and model ancestry."""
        path = self.dataset_provider.config.path
        data_dir = self.dataset_provider.config.data_dir
        if path not in ("parquet", "csv", "json", "tsv"):
            provenance.ancestry.datasets.add(path)
        elif data_dir is not None:
            local = Path(data_dir).expanduser().resolve()
            try:
                provenance.ancestry.datasets.add(f"~/{local.relative_to(Path.home())}")
            except ValueError:
                provenance.ancestry.datasets.add(str(local))

        return provenance
