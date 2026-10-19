import dataclasses
from typing import Any, Literal

from larry.common.config.config import Config
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class PrunerConfig(PreprocessorConfig):
    """Superclass for configs for all ``Pruner`` implementations, configuring any number of input columns."""

    input_columns: list[str] = dataclasses.field(default_factory=list)
    prune_nulls: bool = True


@dataclasses.dataclass(kw_only=True)
class MatchOpConfig(Config):
    values: list[Any] = dataclasses.field(default_factory=list)
    containing: bool = False
    case_sensitive: bool = True
    strip: bool = True


@dataclasses.dataclass(kw_only=True)
class RangeOpConfig(Config):
    min_value: float | None = None
    max_value: float | None = None


@dataclasses.dataclass(kw_only=True)
class ColumnValuesPrunerConfig(PrunerConfig):
    """Config class for ``ColumnValuesPruner``, allowing configuration of slightly complex logic for pruning rows."""

    op: Literal["matches", "nmatches", "in_range", "nin_range"]
    multiple_inputs_behavior: Literal["any", "all"] = "any"
    op_config: MatchOpConfig | RangeOpConfig | None = None

    def __post_init__(self) -> None:
        expected = RangeOpConfig if self.op in ("in_range", "nin_range") else MatchOpConfig
        if self.op_config is None:
            self.op_config = expected()
        if not isinstance(self.op_config, expected):
            raise ValueError(f"op={self.op!r} takes a {expected.__name__}, got {type(self.op_config).__name__}")


@dataclasses.dataclass(kw_only=True)
class CTCMismatchColumnValuesPrunerConfig(ColumnValuesPrunerConfig):
    """Specific config class for the ``ColumnValuesPruner``.

    Configures for a matcher to prune rows where the ctc_mismatch is greater than ``max_value`` in ``op_config``.
    """

    input_columns: list[str] = dataclasses.field(default_factory=lambda: [
        "ctc_mismatch"
    ])
    op: Literal["matches", "nmatches", "in_range", "nin_range"] = "in_range"


@dataclasses.dataclass(kw_only=True)
class ASRCTCMismatchColumnValuesPrunerConfig(CTCMismatchColumnValuesPrunerConfig):
    """Prunes rows where the ctc_mismatch is greater than 0.5."""

    op_config: MatchOpConfig | RangeOpConfig | None = dataclasses.field(
        default_factory=lambda: RangeOpConfig(max_value=0.5)
    )


@dataclasses.dataclass(kw_only=True)
class TTSCTCMismatchColumnValuesPrunerConfig(CTCMismatchColumnValuesPrunerConfig):
    """Prunes rows where the ctc_mismatch is greater than 0.2."""

    op_config: MatchOpConfig | RangeOpConfig | None = dataclasses.field(
        default_factory=lambda: RangeOpConfig(max_value=0.2)
    )
