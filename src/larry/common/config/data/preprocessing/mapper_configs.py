import dataclasses
from typing import Any

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


def _resolve(name: str, configured: list[str], required: list[str]) -> list[str]:
    if not configured:
        return list(dict.fromkeys(required))
    missing = [c for c in required if c not in configured]
    if missing:
        raise ValueError(f"{name}={configured} does not include {missing}")
    return list(dict.fromkeys(configured))


@dataclasses.dataclass(kw_only=True)
class MapperConfig(PreprocessorConfig):
    filter_null_outputs: bool = False
    remove_columns: list[str] = dataclasses.field(default_factory=list)
    input_columns: list[str] = dataclasses.field(default_factory=list)
    output_columns: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.input_columns = _resolve("input_columns", self.input_columns, self.required_input_columns())
        self.output_columns = _resolve("output_columns", self.output_columns, self.required_output_columns())

    def required_input_columns(self) -> list[str]:
        return []

    def required_output_columns(self) -> list[str]:
        return []


@dataclasses.dataclass(kw_only=True)
class SingleColumnMapperConfig(MapperConfig):
    input_column: str
    output_column: str

    def required_input_columns(self) -> list[str]:
        return [self.input_column]

    def required_output_columns(self) -> list[str]:
        return [self.output_column]


@dataclasses.dataclass
class ValueOverride:
    input_column: str
    input_column_value: Any
    output_column: str
    output_column_value: Any


@dataclasses.dataclass(kw_only=True)
class ValueOverrideMapperConfig(MapperConfig):
    mappings_config: list[dict[str, Any]]

    @property
    def mappings(self) -> list[ValueOverride]:
        return [
            ValueOverride(m["input_column"], m["input_column_value"], m["output_column"], m["output_column_value"])
            for m in self.mappings_config
        ]

    def required_input_columns(self) -> list[str]:
        return list(dict.fromkeys(c for m in self.mappings for c in (m.input_column, m.output_column)))

    def required_output_columns(self) -> list[str]:
        return list(dict.fromkeys(m.output_column for m in self.mappings))


@dataclasses.dataclass(kw_only=True)
class UrlMapperConfig(SingleColumnMapperConfig):
    input_column: str = "url"

    timeout: float = 10.0
    busy_wait: float = 0.0
    retry_backoff: float = 1.0
    max_retries: int = 2

    download_cache_dir: str = "/tmp/larry/url_mapper"


@dataclasses.dataclass(kw_only=True)
class UrlBytesMapperConfig(UrlMapperConfig):
    output_column: str = "bytes"
    validate_starts_with: str | None = None

    download_cache_dir: str = "/tmp/larry/bytes_url_mapper"


@dataclasses.dataclass(kw_only=True)
class CompositeIDMapperConfig(MapperConfig):
    id_column: str = "id"

    def required_input_columns(self) -> list[str]:
        return self.input_columns

    def required_output_columns(self) -> list[str]:
        return [self.id_column]
