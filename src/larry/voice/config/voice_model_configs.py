import dataclasses

from larry.common.config.model_config import ModelConfig


@dataclasses.dataclass
class VoiceModelConfig(ModelConfig):
    """Super class for all DTOs related to text model configuration."""
