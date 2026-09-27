import dataclasses

from larry.common.config.preprocessing_mapper_configs import PreprocessingMapperConfig


@dataclasses.dataclass(kw_only=True)
class TextPreprocessingMapperConfig(PreprocessingMapperConfig):
    """Super class for all DTOs related to voice dataset preprocessing."""

    input_column: str = "text"
    output_column: str | list[dict[str, str]] | None = "text"


@dataclasses.dataclass(kw_only=True)
class TextTokenizationMapperConfig(TextPreprocessingMapperConfig):
    """Config for tokenization of text."""

    tokenizer: str

    output_column: str | list[dict[str, str]] | None = "input_ids"


@dataclasses.dataclass(kw_only=True)
class SmolLM2TextTokenizationPreprocessingMapperConfig(TextTokenizationMapperConfig):
    """Config for tokenizing using the SmolLM2 model family's tokenizer."""

    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
