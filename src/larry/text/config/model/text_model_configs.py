import dataclasses

from larry.common.config.model.model_configs import ModelConfig


@dataclasses.dataclass(kw_only=True)
class TextModelConfig(ModelConfig):
    """Superclass for all text model configs. Adds sensible required fields for any model concerned with text."""

    tokenizer_name: str
    vocab_size: int
