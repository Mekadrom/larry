import copy
import dataclasses
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.common.utils import git_utils, load_yaml_dict

log = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class ProvenanceAncestry:
    """Helper dataclass for storing provenance ancestry - immediate source datasets and models.

    Shared by preprocessing and training pipelines.
    """

    datasets: set[str]
    models: set[str]

    @classmethod
    def new_ancestry(cls, dictionary: dict[str, Any]) -> Self:
        """Instantiates a new instance of ``ProvenanceAncestry`` using the provided ``dict``."""
        datasets = dictionary.get("datasets", set())
        models = dictionary.get("models", set())
        return cls(datasets=datasets, models=models)

    @classmethod
    def from_dict(cls, a_dict: dict[str, Any]) -> Self:
        """Creates a new ``ProvenanceAncestry`` instance using the provided dictionary as field values."""
        datasets = a_dict.get("datasets")
        if datasets is None:
            datasets = set()

        datasets = set(datasets)

        models = a_dict.get("models")
        if models is None:
            models = set()

        models = set(models)
        return cls(datasets=datasets, models=models)


@dataclasses.dataclass(kw_only=True)
class Provenance:
    """Helper dataclass for tracking the provenance of outputs of either preprocessing or training pipelines."""

    id_columns: list[str]
    ancestry: ProvenanceAncestry
    produced_on: str | None = None
    git_commit: str | None = None
    git_is_dirty: bool | None = None

    @classmethod
    def new_provenance(cls, dictionary: dict[str, Any], ancestry: ProvenanceAncestry | None = None) -> Self:
        """Instantiates a new instance of ``Provenance`` using the provided ``dict``. Takes an ancestry override."""
        copy_dict = copy.deepcopy(dictionary)
        if not ancestry:
            ancestry = ProvenanceAncestry.from_dict(copy_dict.pop("ancestry", {}))
        else:
            copy_dict.pop("ancestry", None)

        id_columns = copy_dict.get("id_columns", [])

        return cls(
            id_columns=id_columns,
            ancestry=ancestry
        )

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
class PipelineConfig:
    """Superclass for configuring the entire pipeline."""

    parquet_size_mb: int = 500
    default_batch_size: int = 1000
    seed: int = 42

    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)


class PipelineState:
    """Delegate for pipeline and preprocessing config parsing functionality and state storage for a run."""

    config: PipelineConfig
    dataset_provider: DatasetProvider
    pipeline: list[Preprocessor]
    provenance: Provenance

    def __init__(self, config_file: Path) -> None:
        """Parses yaml config into dictionaries and dictionaries into config objects."""
        self.log = logging.getLogger(__name__)

        self.log.info(f"Loading pipeline config from yaml config_file={config_file}")
        definition = load_yaml_dict(config_file)
        self.log.info("Pipeline config loaded.")

        preprocessor_config_dict: dict[str, Any] = definition.get("preprocessor", {})
        preprocessor_inputs: dict[str, Any] = preprocessor_config_dict.get("inputs", [])
        self.log.info(f"using preprocessor_inputs={preprocessor_inputs}")
        self.config = PipelineConfig(**preprocessor_inputs)

        dataset_config_dict: dict[str, Any] = definition.get("dataset", {})

        dataset_config_type_name = dataset_config_dict.get("config")
        dataset_config_split_fractions = dataset_config_dict.get("splits")
        self.fractions: dict[str, float] | None = None
        if dataset_config_split_fractions:
            self.fractions = {i["name"]: i["part"] for i in dataset_config_split_fractions}
            if not math.isclose(sum(self.fractions.values()), 1.0, abs_tol=1e-6):
                raise ValueError(f"split fractions must sum to 1, got {self.fractions}")

        dataset_config_type = DatasetConfig.REGISTRY.resolve(dataset_config_type_name)
        self.log.info(f"Using dataset_config_type_name={dataset_config_type_name}")

        dataset_config_overrides = dataset_config_dict.get("overrides", [])
        self.log.info(f"Applying dataset_config_overrides={dataset_config_overrides}")

        dataset_provider_config: DatasetConfig = dataset_config_type(**dataset_config_overrides)

        dataset_provider_type_name = dataset_provider_config.provider_name
        self.log.info(
            f"Using dataset_provider_type_name={dataset_provider_type_name} "
            f"for dataset_provider_config={dataset_provider_config}"
        )
        dataset_provider_type = DatasetProvider.REGISTRY.resolve(dataset_provider_type_name)
        self.dataset_provider = dataset_provider_type(dataset_provider_config)

        self.provenance = self._build_provenance(copy.deepcopy(self.config.provenance))

        pipeline_configs = definition.get("pipeline", None)
        if pipeline_configs:
            self.pipeline = self._build_mapper_pipeline(pipeline_configs)
        else:
            self.log.critical(
                "No pipeline configured; parquets will be downloaded and cached locally according to the configuration."
            )
            self.pipeline = []
        self.log.info("Preprocessor constructed")

    def _build_mapper_pipeline(self, pipeline_configs: list[dict[str, Any]]) -> list[Preprocessor]:
        """Builds preprocessor instances in order using defined configs and config overrides."""
        pipeline: list[Preprocessor] = []

        for preprocessor_config in pipeline_configs:
            preprocessor_type_name = preprocessor_config.get("name")
            preprocessor_type = Preprocessor.REGISTRY.resolve(preprocessor_type_name)

            preprocessor_config_type_name = preprocessor_config.get("config")
            preprocessor_config_type = PreprocessorConfig.REGISTRY.resolve(preprocessor_config_type_name)

            self.log.info(
                f"Using preprocessor_config_type_name={preprocessor_config_type_name} "
                f"for preprocessor_type_name={preprocessor_type_name}"
            )

            preprocessor_config_overrides = preprocessor_config.get("inputs", {})
            self.log.info(f"Applying preprocessor_config_overrides={preprocessor_config_overrides}")
            preprocessor_config = preprocessor_config_type(**preprocessor_config_overrides)

            self.log.info(
                f"preprocessor_type_name={preprocessor_type_name} "
                f"using preprocessor_config={preprocessor_config}"
            )

            pipeline.append(preprocessor_type(self.provenance, preprocessor_config))

        return pipeline

    def _build_provenance(self, p_dict: dict[str, Any]) -> Provenance:
        """Builds provenance using intuited defaults for dataset and model ancestry."""
        ancestry = ProvenanceAncestry.from_dict(p_dict.pop("ancestry", {}))

        path = self.dataset_provider.config.path
        data_dir = self.dataset_provider.config.data_dir
        if path not in ("parquet", "csv", "json", "tsv"):
            ancestry.datasets.add(path)
        elif data_dir is not None:
            local = Path(data_dir).expanduser().resolve()
            try:
                ancestry.datasets.add(f"~/{local.relative_to(Path.home())}")
            except ValueError:
                ancestry.datasets.add(str(local))

        return Provenance.new_provenance(p_dict, ancestry)
