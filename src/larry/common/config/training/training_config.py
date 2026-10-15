import dataclasses
import logging
from pathlib import Path
from typing import Literal, Any

from accelerate import PartialState
from datasets.naming import camelcase_to_snakecase

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.training import build_trainer_config, build_model_config
from larry.common.metrics.metrics import WandBMetrics, TensorBoardMetrics, Metrics, NoneMetrics
from larry.common.training import trainers
from larry.common.training.training_model import LarryModel
from larry.common.utils import load_yaml_dict
from larry.common.utils.types import FactoryRegistry

_metrics: FactoryRegistry[Metrics] = {
    "wandb": WandBMetrics,
    "tensorboard": TensorBoardMetrics,
    "tb": TensorBoardMetrics,
}


@dataclasses.dataclass(kw_only=True)
class TrainingRunConfig:
    """Config for the entire training pipeline."""
    metrics_backend: Literal["wandb", "tensorboard", "db"] = "wandb"

    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)


class TrainingRunState:
    config: TrainingRunConfig

    def __init__(self, config_file: Path, log_dir: Path, resume_dir: str | None = None) -> None:
        self.log = logging.getLogger(type(self).__name__)

        self.log.info(f"Loading pipeline file from yaml config_file={config_file}")
        definition = load_yaml_dict(config_file)
        self.log.info("Training config loaded.")

        training_config_dict: dict[str, Any] = definition.get("training", {})
        metrics_backend = training_config_dict.get("metricsBackend", "tensorboard")
        self.log.info(f"Using {metrics_backend} metrics backend.")

        training_config_dict = {camelcase_to_snakecase(k): v for k, v in training_config_dict.items()}

        self.config = TrainingRunConfig(**training_config_dict)

        trainer_config_dict = definition.get("trainer", {})
        trainer_type_name = trainer_config_dict.get("name")

        model_config_dict: dict[str, Any] = definition.get("model", {})
        model_type_name = model_config_dict.get("name")

        self.trainer_config = build_trainer_config(trainer_config_dict)
        model_config = build_model_config(model_config_dict)

        state = PartialState()
        metrics = _metrics[metrics_backend](log_dir) if state.is_main_process else NoneMetrics(log_dir=log_dir)

        model_type = LarryModel.REGISTRY.resolve(model_type_name)
        model = model_type(model_config)

        trainer_type = trainers.TrainerBase.REGISTRY.resolve(trainer_type_name)
        provenance = Provenance.new_provenance(training_config_dict.get("provenance", {}))

        self.trainer = trainer_type(provenance, model, self.trainer_config, metrics, resume_dir=resume_dir)

        self.log.info(f"trainer={self.trainer}")
        self.log.info(f"trainer_config={self.trainer_config}")
