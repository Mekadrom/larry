import dataclasses

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig


@dataclasses.dataclass(kw_only=True)
class TextMapperConfig(SingleColumnMapperConfig):
    input_column: str = "text"
    output_column: str = "text"


@dataclasses.dataclass(kw_only=True)
class UrlTextMapperConfig(UrlMapperConfig, TextMapperConfig):
    input_column: str = "url"
    output_column: str = "text"

    download_cache_dir: str = "/tmp/larry/text_url_mapper"

    hash_column: str = "text_url_hash"


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptPdfTextMapperConfig(SingleColumnMapperConfig):
    input_column: str = "bytes_url_hash"
    docket_column: str = "docket"
    output_column: str = "transcript"

    download_cache_dir: str = "/tmp/larry/bytes_url_mapper"


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
