import dataclasses

from transformers import AutoModel


@dataclasses.dataclass
class ModelConfig:
    """Base class for all config DTOs having to do with a model; no functionality."""

    auto_model_type: AutoModel
    path: str | None = None
