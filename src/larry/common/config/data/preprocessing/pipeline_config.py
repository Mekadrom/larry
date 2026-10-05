import dataclasses


@dataclasses.dataclass(kw_only=True)
class PipelineConfig:
    """Config for an entire run of preprocessing."""

    parquet_size_mb: int = 500
    provenance_columns: list[str] = dataclasses.field(default_factory=list)
