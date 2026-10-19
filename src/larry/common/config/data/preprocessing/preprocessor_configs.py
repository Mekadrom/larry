import dataclasses

from larry.common.config.config import Config, TaggedConfig


@dataclasses.dataclass(kw_only=True)
class PreprocessorConfig(TaggedConfig):
    """Superclass for configs for any step in a preprocessing pipeline for datasets."""

    name: str = ""

    batch_size: int = 1000
    writer_batch_size: int = 1000
    num_proc: int = 8
    cache_results: bool = True


@dataclasses.dataclass(kw_only=True)
class RemoveColumnsPreprocessorConfig(PreprocessorConfig):
    """Config for the ``RemoveColumnsPreprocessor``, allowing configuration of multiple columns to drop."""

    remove_columns: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class JsonPathMapping(Config):
    json_path: str
    output_column: str


@dataclasses.dataclass(kw_only=True)
class NestedExtractionPreprocessorConfig(PreprocessorConfig):
    """Config for the ``NestedExtractionPreprocessor``, allowing configuration of json queries for extracted data.

    Also includes configuration of columns to remove after preprocessing.
    """

    input_column: str
    remove_columns: list[str] = dataclasses.field(default_factory=list)
    json_path_mappings: list[JsonPathMapping] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class ColumnCastingPreprocessorConfig(PreprocessorConfig):
    """Config for the ``ColumnCastingPreprocessor``, allowing configuration of column cast mappings."""

    column_casts: dict[str, str | None] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class ColumnRenamingPreprocessorConfig(PreprocessorConfig):
    """Config for the ``ColumnRenamingPreprocessor``, allowing configuration of column name mappings."""

    column_name_mappings: dict[str, str | None] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class RowIndexAssigningPreprocessorConfig(PreprocessorConfig):
    """Config for the ``RowIndexAssigningPreprocessor``."""

    output_column: str = "row"
