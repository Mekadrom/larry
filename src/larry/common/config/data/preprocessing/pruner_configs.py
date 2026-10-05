import dataclasses
from typing import Any, Literal

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class PrunerConfig(PreprocessorConfig):
    prune_nulls: bool = True


@dataclasses.dataclass(kw_only=True)
class ColumnValuesPrunerConfig(PrunerConfig):
    op: Literal["matches", "nmatches"]
    values: list[Any | None] = dataclasses.field(default_factory=list)
    op_config: dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "containing": False,
        "case_insensitive": True,
        "strip": True,
    })
