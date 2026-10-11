import dataclasses

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig, \
    MapperConfig


@dataclasses.dataclass(kw_only=True)
class UrlTextMapperConfig(UrlMapperConfig, SingleColumnMapperConfig):
    input_column: str = "url"
    output_column: str = "text"

    download_cache_dir: str = "/tmp/larry/text_url_mapper"


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptPdfTextMapperConfig(MapperConfig):
    pdf_bytes_column: str = "pdf_bytes"
    docket_column: str = "docket"
    date_argued_column: str = "date_argued"

    transcript_output_column: str = "transcript"
    reattributions_output_column: str = "reattributions"

    def required_input_columns(self) -> list[str]:
        return [self.pdf_bytes_column, self.docket_column, self.date_argued_column]

    def required_output_columns(self) -> list[str]:
        return [self.transcript_output_column, self.reattributions_output_column]


@dataclasses.dataclass(kw_only=True)
class TextTokenizingMapperConfig(SingleColumnMapperConfig):
    tokenizer: str
    input_column: str = "text"
    output_column: str = "input_ids"


@dataclasses.dataclass(kw_only=True)
class TinyShakespeareTextTokenizingMapperConfig(TextTokenizingMapperConfig):
    tokenizer: str = "google/byt5-small"


@dataclasses.dataclass(kw_only=True)
class SmolLM2TextTokenizingMapperConfig(TextTokenizingMapperConfig):
    tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
