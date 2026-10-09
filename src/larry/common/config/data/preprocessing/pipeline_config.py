import dataclasses
from typing import Any


@dataclasses.dataclass(kw_only=True)
class PipelineConfig:
    """Config for an entire run of preprocessing."""

    parquet_size_mb: int = 500
    provenance_columns: list[str] = dataclasses.field(default_factory=list)
    default_batch_size: int = 1000
    seed: int = 42

    provenance_overrides: list[dict[str, list[Any]]] = dataclasses.field(default_factory=list)
