import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
    metrics_backend: Literal["wandb", "tensorboard", "db"] = "wandb"
