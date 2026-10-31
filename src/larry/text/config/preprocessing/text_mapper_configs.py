import dataclasses

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig, SingleColumnMapperConfig, UrlMapperConfig


@dataclasses.dataclass(kw_only=True)
class UrlTextMapperConfig(UrlMapperConfig):
    """Config for ``UrlTextMapper`` instances, setting sensible defaults for fields in ``UrlMapperConfig``."""

    input_column: str = "url"
    output_column: str = "text"

    download_cache_dir: str = "/tmp/larry/text_url_mapper"


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptPdfTextMapperConfig(MapperConfig):
    """Config for ``SCOTUSTranscriptPdfTextMapper`` instances, setting defaults and adding new fields and constants."""

    pdf_bytes_column: str = "pdf_bytes"
    docket_column: str = "docket"
    date_argued_column: str = "date_argued"

    transcript_output_column: str = "transcript"
    reattributions_output_column: str = "reattributions"

    def required_input_columns(self) -> list[str]:
        """Overrides to set the input columns this mapper requires."""
        return [self.pdf_bytes_column, self.docket_column, self.date_argued_column]

    def required_output_columns(self) -> list[str]:
        """Overrides to set the expected output columns this mapper produces."""
        return [self.transcript_output_column, self.reattributions_output_column]


@dataclasses.dataclass(kw_only=True)
class TextTokenizingMapperConfig(SingleColumnMapperConfig):
    """Config for ``TextTokenizingMapper`` instances, setting sensible defaults and requiring a tokenizer value."""

    tokenizer: str
    input_column: str = "text"
    output_column: str = "input_ids"


@dataclasses.dataclass(kw_only=True)
class VisualDescriptionExtractingMapperConfig(MapperConfig):
    input_column: str = "description"
    output_column: str = "visual_description"
    retained_output_column: str = "visual_retained"

    paragraph_threshold: float = 0.0
    sentence_drop_threshold: float = -0.15
    sentence_keep_threshold: float = 0.15
    extra_visual_terms: list[str] = dataclasses.field(default_factory=list)
    extra_history_terms: list[str] = dataclasses.field(default_factory=list)

    def required_input_columns(self) -> list[str]:
        return [self.input_column]

    def required_output_columns(self) -> list[str]:
        return [self.output_column, self.retained_output_column]


@dataclasses.dataclass(kw_only=True)
class CaptionSelectingMapperConfig(MapperConfig):
    caption_columns: list[str]
    output_column: str = "caption"
    source_output_column: str = "caption_source"
    min_characters: int = 1
    item_key: str = "caption"

    def required_input_columns(self) -> list[str]:
        return list(self.caption_columns)

    def required_output_columns(self) -> list[str]:
        return [self.output_column, self.source_output_column]
