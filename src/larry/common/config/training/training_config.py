import dataclasses
from typing import Literal, Any


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
    metrics_backend: Literal["wandb", "tensorboard", "db"] = "wandb"

    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)
