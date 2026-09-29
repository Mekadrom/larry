import dataclasses

from larry.common.config.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class MapperConfig(PreprocessorConfig):
    output_column: str | None = None
    remove_columns: list[str] | None = dataclasses.field(default_factory=list)
    filter_null_outputs: bool = False
