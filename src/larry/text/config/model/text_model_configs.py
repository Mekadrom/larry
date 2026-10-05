import dataclasses

from larry.common.config.model.model_configs import ModelConfig


@dataclasses.dataclass(kw_only=True)
class TextModelConfig(ModelConfig):
    tokenizer_name: str
    vocab_size: int


@dataclasses.dataclass(kw_only=True)
class SmokeTestLLMModelConfig(TextModelConfig):
    tokenizer_name: str = "google/byt5-small"
    vocab_size: int = 384
