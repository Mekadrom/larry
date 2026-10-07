import dataclasses
from typing import Any

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class MapperConfig(PreprocessorConfig):
    remove_columns: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class SingleColumnMapperConfig(MapperConfig):
    input_column: str
    output_column: str
    filter_null_outputs: bool = False
    keep_input: bool = False


@dataclasses.dataclass
class ValueOverride:
    input_column: str
    input_column_value: Any
    output_column: str
    output_column_value: Any


@dataclasses.dataclass(kw_only=True)
class ValueOverrideMapperConfig(MapperConfig):
    mappings_in: list[dict[str, Any]]

    @property
    def mappings(self) -> list[ValueOverride]:
        return [
            ValueOverride(m["input_column"], m["input_column_value"], m["output_column"], m["output_column_value"])
            for m in self.mappings_in
        ]


@dataclasses.dataclass(kw_only=True)
class UrlMapperConfig(SingleColumnMapperConfig):
    input_column: str = "url"

    timeout: float = 10.0
    busy_wait: float = 0.0
    retry_backoff: float = 1.0
    max_retries: int = 2

    download_cache_dir: str = "/tmp/larry/url_mapper"

    hash_column: str = "url_hash"


@dataclasses.dataclass(kw_only=True)
class UrlBytesMapperConfig(UrlMapperConfig):
    output_column: str = "bytes"
    validate_starts_with: str | None = None

    download_cache_dir: str = "/tmp/larry/bytes_url_mapper"

    hash_column: str = "bytes_url_hash"
