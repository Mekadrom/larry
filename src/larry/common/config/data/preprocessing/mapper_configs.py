import dataclasses
from typing import Any

from larry.common.config.config import Config
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class MapperConfig(PreprocessorConfig):
    """Superclass for all ``Mapper`` config classes.

    Configures at least one input column and at least one output column, as well as the optional functionality to
    configure columns to remove and whether to filter null outputs from the mapper step.
    """

    filter_null_outputs: bool = False
    remove_columns: list[str] = dataclasses.field(default_factory=list)
    input_columns: list[str] = dataclasses.field(default_factory=list)
    output_columns: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Required to ensure that the backing fields are in sync with the potentially overridden methods."""
        self.input_columns = self._resolve("input_columns", self.input_columns, self.required_input_columns())
        self.output_columns = self._resolve("output_columns", self.output_columns, self.required_output_columns())

    def required_input_columns(self) -> list[str]:
        """Returns all required input columns."""
        return []

    def required_output_columns(self) -> list[str]:
        """Returns all required output columns."""
        return []

    @staticmethod
    def _resolve(name: str, configured: list[str], required: list[str]) -> list[str]:
        """Ensures that required columns provides at most the columns that are initialized from overrides."""
        if not configured:
            return list(dict.fromkeys(required))

        missing = [c for c in required if c not in configured]
        if missing:
            raise ValueError(f"{name}={configured} does not include {missing}")

        return list(dict.fromkeys(configured))


@dataclasses.dataclass(kw_only=True)
class SingleColumnMapperConfig(MapperConfig):
    """Superclass config class for all ``Mapper`` subclasses that only require one input and one output column."""

    input_column: str
    output_column: str

    def required_input_columns(self) -> list[str]:
        """Returns the single input column."""
        return [self.input_column]

    def required_output_columns(self) -> list[str]:
        """Returns the single output column."""
        return [self.output_column]


@dataclasses.dataclass(kw_only=True)
class ValueOverride(Config):
    """Config helper dataclass for type hinting."""

    input_column: str
    input_column_value: Any
    output_column: str
    output_column_value: Any


@dataclasses.dataclass(kw_only=True)
class ValueOverrideMapperConfig(MapperConfig):
    """Config class for the ``ValueOverrideMapper``."""

    mappings: list[ValueOverride]

    def required_input_columns(self) -> list[str]:
        """Dynamically assigns input columns as both the input columns and output columns (for convenience)."""
        return list(dict.fromkeys(c for m in self.mappings for c in (m.input_column, m.output_column)))

    def required_output_columns(self) -> list[str]:
        """Returns the output columns from the mapping configs."""
        return list(dict.fromkeys(m.output_column for m in self.mappings))


@dataclasses.dataclass(kw_only=True)
class UrlMapperConfig(SingleColumnMapperConfig):
    """Superclass config for all ``UrlMapper`` implementations.

    Configures a default url input column and download configs.
    """

    input_column: str = "url"

    timeout: float = 10.0
    busy_wait: float = 0.0
    retry_backoff: float = 1.0
    max_retries: int = 2

    download_cache_dir: str = "/tmp/larry/url_mapper"


@dataclasses.dataclass(kw_only=True)
class UrlBytesMapperConfig(UrlMapperConfig):
    """Subclass config for ``UrlByteMapper`` that configures additional defaults."""

    output_column: str = "bytes"
    validate_starts_with: str | None = None

    download_cache_dir: str = "/tmp/larry/bytes_url_mapper"


@dataclasses.dataclass(kw_only=True)
class CompositeIDMapperConfig(MapperConfig):
    """Config class for the ``CompositeIDMapper``."""

    id_output_column: str = "id"
    delim: str = ":"

    def required_input_columns(self) -> list[str]:
        """Returns all configured input columns."""
        return self.input_columns

    def required_output_columns(self) -> list[str]:
        """Returns the single id output column."""
        return [self.id_output_column]
