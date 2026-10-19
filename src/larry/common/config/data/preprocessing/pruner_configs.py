import dataclasses
from typing import Any, Literal

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class PrunerConfig(PreprocessorConfig):
    """Superclass for configs for all ``Pruner`` implementations, configuring any number of input columns."""

    input_columns: list[str] = dataclasses.field(default_factory=list)
    prune_nulls: bool = True


@dataclasses.dataclass(kw_only=True)
class ColumnValuesPrunerConfig(PrunerConfig):
    """Config class for ``ColumnValuesPruner``, allowing configuration of slightly complex logic for pruning rows."""

    op: Literal["matches", "nmatches", "in_range", "nin_range"]
    multiple_inputs_behavior: Literal["any", "all"] = "any"
    op_config: dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "containing": False,
        "case_sensitive": False,
        "strip": True,
    })


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

    op_config: dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "max_value": 0.5
    })


@dataclasses.dataclass(kw_only=True)
class TTSCTCMismatchColumnValuesPrunerConfig(CTCMismatchColumnValuesPrunerConfig):
    """Prunes rows where the ctc_mismatch is greater than 0.2."""

    op_config: dict[str, Any] = dataclasses.field(default_factory=lambda: {
        "max_value": 0.2
    })
