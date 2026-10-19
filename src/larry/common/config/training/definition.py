import dataclasses
from typing import Any, Literal

from larry.common.config.config import Config
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig


@dataclasses.dataclass(kw_only=True)
class TrainingRunConfig(Config):
    """Config for the entire training pipeline."""

    metrics_backend: Literal["wandb", "tensorboard", "tb"] = "tensorboard"

    provenance: Provenance = dataclasses.field(default_factory=Provenance)


@dataclasses.dataclass(kw_only=True)
class TrainingDefinition(Config):
    training: TrainingRunConfig = dataclasses.field(default_factory=TrainingRunConfig)
    trainer: LarryTrainerConfig[Any]
    model: ModelConfig

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        return {**d, "trainer": self.trainer.to_dict(), "model": self.model.to_dict()}
