import dataclasses
from typing import Any, Literal


@dataclasses.dataclass(kw_only=True)
class PreprocessingMapperConfig:
    """Base class for all config DTOs having to do with a preprocessing run; no functionality."""

    input_column: str | None = None
    output_column: str | list[dict[str, str]] | None = None
    remove_columns: list[str] | None = dataclasses.field(default_factory=list)
    batched: bool = True
    batch_size: int = 2000
    num_proc: int = 8
    cache_results: bool = True

    def load_overrides(self, config_dict: dict[str, Any]) -> None:
        [setattr(self, k, v) for k, v in config_dict.items()]


@dataclasses.dataclass(kw_only=True)
class PruningPreprocessingMapperConfig(PreprocessingMapperConfig):
    remove_column: bool = False
    drop_missing: bool = True


def default_op_config() -> dict[str, Any]:
    return {
        "containing": False,
        "case_insensitive": True,
        "strip": True,
    }


@dataclasses.dataclass(kw_only=True)
class ColumnValuesPruningPreprocessingMapperConfig(PruningPreprocessingMapperConfig):
    op: str
    values: list[Any | None] = dataclasses.field(default_factory=list)
    op_config: dict[str, Any] = dataclasses.field(default_factory=default_op_config)


@dataclasses.dataclass(kw_only=True)
class NestedExtractionPreprocessingMapperConfig(PreprocessingMapperConfig):
    json_path_to_output_column_mappings: list[dict[str, str]] = dataclasses.field(default_factory=list)
    drop_missing_type: Literal["any", "all", "never"] = "any"
