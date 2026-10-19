import copy
import dataclasses
import logging
from pathlib import Path

from accelerate import PartialState

from larry.common.config.training.definition import TrainingDefinition, TrainingRunConfig
from larry.common.config.values import RunRecord
from larry.common.metrics.metrics import Metrics, NoneMetrics, TensorBoardMetrics, WandBMetrics
from larry.common.training import trainers
from larry.common.training.training_model import LarryModel
from larry.common.utils.types import FactoryRegistry

_metrics: FactoryRegistry[Metrics] = {
    "wandb": WandBMetrics,
    "tensorboard": TensorBoardMetrics,
    "tb": TensorBoardMetrics,
}


class TrainingRunState:
    """Helper class for parsing the config for a training run."""

    config: TrainingRunConfig

    def __init__(self, definition: TrainingDefinition, log_dir: Path, resume_dir: str | None = None) -> None:
        self.log = logging.getLogger(type(self).__name__)

        self.definition = definition
        self.config = definition.training
        self.trainer_config = definition.trainer
        model_config = definition.model
        self.log.info(f"Using {self.config.metrics_backend} metrics backend.")

        state = PartialState()
        metrics_type = _metrics[self.config.metrics_backend]
        metrics = metrics_type(log_dir) if state.is_main_process else NoneMetrics(log_dir=str(log_dir))

        model_type = LarryModel.REGISTRY.for_config(model_config)
        model = model_type(model_config)

        trainer_type = trainers.TrainerBase.REGISTRY.for_config(self.trainer_config)
        record = RunRecord.of(definition)
        provenance = dataclasses.replace(copy.deepcopy(self.config.provenance), config=record.config, cli=record.cli)

        self.trainer = trainer_type(provenance, model, self.trainer_config, metrics, resume_dir=resume_dir)

        self.log.info(f"trainer={self.trainer}")
        self.log.info(f"trainer_config={self.trainer_config}")
