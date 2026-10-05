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
