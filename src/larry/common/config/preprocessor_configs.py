import dataclasses
from typing import Any


@dataclasses.dataclass(kw_only=True)
class PreprocessorConfig:
    """Base class for all config DTOs having to do with a preprocessing run; no functionality."""

    input_column: str | None = None
    batch_size: int = 2000
    num_proc: int = 8
    cache_results: bool = True

    def load_overrides(self, config_dict: dict[str, Any]) -> None:
        [setattr(self, k, v) for k, v in config_dict.items()]


@dataclasses.dataclass(kw_only=True)
class RemoveColumnsPreprocessorConfig(PreprocessorConfig):
    remove_columns: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class NestedExtractionPreprocessorConfig(PreprocessorConfig):
    remove_columns: list[str] | None = dataclasses.field(default_factory=list)
    json_path_to_output_column_mappings: list[dict[str, str]] = dataclasses.field(default_factory=list)
