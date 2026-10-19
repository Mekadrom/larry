import dataclasses
import logging
from pathlib import Path
from typing import Any, Literal

from accelerate import PartialState

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.training import build_model_config, build_trainer_config
from larry.common.metrics.metrics import Metrics, NoneMetrics, TensorBoardMetrics, WandBMetrics
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

    metrics_backend: Literal["wandb", "tensorboard", "tb"] = "tensorboard"

    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)


class TrainingRunState:
    """Helper class for parsing the config for a training run."""

    config: TrainingRunConfig

    def __init__(self, config_file: Path, log_dir: Path, resume_dir: str | None = None) -> None:
        """Parses the specified config file into a ``TrainingRunConfig`` object. Also instantiates the trainer."""
        self.log = logging.getLogger(type(self).__name__)

        self.log.info(f"Loading pipeline file from yaml config_file={config_file}")
        definition = load_yaml_dict(config_file)
        self.log.info("Training config loaded.")

        self.config = TrainingRunConfig(**definition.get("training", {}))
        self.log.info(f"Using {self.config.metrics_backend} metrics backend.")

        trainer_config_dict = definition.get("trainer", {})
        trainer_type_name = trainer_config_dict.get("name")

        model_config_dict: dict[str, Any] = definition.get("model", {})
        model_type_name = model_config_dict.get("name")

        self.trainer_config = build_trainer_config(trainer_config_dict)
        model_config = build_model_config(model_config_dict)

        state = PartialState()
        metrics_type = _metrics[self.config.metrics_backend]
        metrics = metrics_type(log_dir) if state.is_main_process else NoneMetrics(log_dir=log_dir)

        model_type = LarryModel.REGISTRY.resolve(model_type_name)
        model = model_type(model_config)

        trainer_type = trainers.TrainerBase.REGISTRY.resolve(trainer_type_name)
        provenance = Provenance.new_provenance(self.config.provenance)

        self.trainer = trainer_type(provenance, model, self.trainer_config, metrics, resume_dir=resume_dir)

        self.log.info(f"trainer={self.trainer}")
        self.log.info(f"trainer_config={self.trainer_config}")
