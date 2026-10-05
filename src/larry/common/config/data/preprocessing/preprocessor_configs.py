import dataclasses
from typing import ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class PreprocessorConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[PreprocessorConfig]]

    batch_size: int = 1000
    writer_batch_size: int = 1000
    num_proc: int = 8
    cache_results: bool = True


@dataclasses.dataclass(kw_only=True)
class RemoveColumnsPreprocessorConfig(PreprocessorConfig):
    input_column: str
    remove_columns: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class NestedExtractionPreprocessorConfig(PreprocessorConfig):
    input_column: str
    remove_columns: list[str] = dataclasses.field(default_factory=list)
    json_path_to_output_column_mappings: list[dict[str, str]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class ValidationSetSplittingMapperConfig(PreprocessorConfig):
    input_column: str
    val_split_percent: float
