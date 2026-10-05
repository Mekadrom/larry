import dataclasses

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class MapperConfig(PreprocessorConfig):
    output_column: str
    remove_columns: list[str] = dataclasses.field(default_factory=list)
    filter_null_outputs: bool = False
