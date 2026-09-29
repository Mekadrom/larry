import dataclasses
from typing import Any

from larry.common.config.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class PrunerConfig(PreprocessorConfig):
    prune_nulls: bool = True


@dataclasses.dataclass(kw_only=True)
class ColumnValuesPrunerConfig(PrunerConfig):
    op: str
    values: list[Any | None] = dataclasses.field(default_factory=list)
    op_config: dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "containing": False,
        "case_insensitive": True,
        "strip": True,
    })
