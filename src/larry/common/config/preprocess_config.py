import dataclasses
from typing import Any, Self


@dataclasses.dataclass(kw_only=True)
class PreprocessConfig:
    """Config for an entire run of preprocessing."""

    parquet_size_mb: int = 500
    provenance_columns: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Converts from camel-case-keyed dict to this dataclass."""
        keys = {
            "isBatched": "is_batched",
            "batchSize": "batch_size",
            "parquetSizeMB": "parquet_size_mb",
            "provenanceColumns": "provenance_columns",
        }
        return cls(**{field: d[key] for key, field in keys.items() if key in d})
