import dataclasses

from larry.common.config.mapper_configs import MapperConfig


@dataclasses.dataclass(kw_only=True)
class TextMapperConfig(MapperConfig):
    """Super class for all DTOs related to voice dataset preprocessing."""

    input_column: str = "text"
    output_column: str | None = "text"


@dataclasses.dataclass(kw_only=True)
class TextTokenizingMapperConfig(TextMapperConfig):
    """Config for tokenization of text."""

    tokenizer: str
    output_column: str | None = "input_ids"


@dataclasses.dataclass(kw_only=True)
class SmolLM2TextTokenizingMapperConfig(TextTokenizingMapperConfig):
    """Config for tokenizing using the SmolLM2 model family's tokenizer."""

    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
