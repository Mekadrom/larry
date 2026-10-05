import dataclasses

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig


@dataclasses.dataclass(kw_only=True)
class TextMapperConfig(MapperConfig):
    input_column: str = "text"
    output_column: str = "text"


@dataclasses.dataclass(kw_only=True)
class TextTokenizingMapperConfig(TextMapperConfig):
    tokenizer: str
    output_column: str = "input_ids"


@dataclasses.dataclass(kw_only=True)
class TinyShakespeareTextTokenizingMapperConfig(TextTokenizingMapperConfig):
    tokenizer: str = "google/byt5-small"


@dataclasses.dataclass(kw_only=True)
class SmolLM2TextTokenizingMapperConfig(TextTokenizingMapperConfig):
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
