import dataclasses
from typing import Any, ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class PreprocessorConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[PreprocessorConfig]]

    input_column: str
    batch_size: int = 2000
    num_proc: int = 8
    cache_results: bool = True


@dataclasses.dataclass(kw_only=True)
class RemoveColumnsPreprocessorConfig(PreprocessorConfig):
    remove_columns: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class NestedExtractionPreprocessorConfig(PreprocessorConfig):
    remove_columns: list[str] = dataclasses.field(default_factory=list)
    json_path_to_output_column_mappings: list[dict[str, str]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class ValidationSetSplittingMapperConfig(PreprocessorConfig):
    val_split_percent: float
