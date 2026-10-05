import dataclasses

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig


@dataclasses.dataclass(kw_only=True)
class TextSingleColumnMapperConfig(SingleColumnMapperConfig):
    input_column: str = "text"
    output_column: str = "text"


@dataclasses.dataclass(kw_only=True)
class UrlTextMapperConfig(UrlMapperConfig, TextSingleColumnMapperConfig):
    input_column: str = "url"
    output_column: str = "text"

    download_cache_dir: str = "/tmp/larry/text_url_mapper"


@dataclasses.dataclass(kw_only=True)
class TextTokenizingMapperConfig(TextSingleColumnMapperConfig):
    tokenizer: str
    output_column: str = "input_ids"


@dataclasses.dataclass(kw_only=True)
class TinyShakespeareTextTokenizingMapperConfig(TextTokenizingMapperConfig):
    tokenizer: str = "google/byt5-small"


@dataclasses.dataclass(kw_only=True)
class SmolLM2TextTokenizingMapperConfig(TextTokenizingMapperConfig):
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
