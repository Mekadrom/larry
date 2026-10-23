import dataclasses
from typing import Any

from larry.common.config.config import Config, NamedEntries


@dataclasses.dataclass(kw_only=True)
class DatasetSourceConfig(Config):
    """A config dataclass for configuring a single dataset source."""

    name: str
    index_path: str  # huggingface repo or local dir: rows with id, duration_s, text_original, speaker_id, etc.
    provenance_name: str | None = None
    source_path: str | None = None
    source_name: str | None = None
    source_revision: str | None = None
    source_cache_dir: str | None = None
    split: str = "train"
    source_split: str = "train"
    weight: float = 1.0  # target share of audio time, not of rows


@dataclasses.dataclass(kw_only=True)
class TextSourceConfig(DatasetSourceConfig):
    text_column: str = "text"


@dataclasses.dataclass(kw_only=True)
class DatasetMixtureConfig[T: DatasetSourceConfig = DatasetSourceConfig](Config):
    """A config dataclass for defining the entire training and eval mixture of dataset sources and batch hyperparams."""

    train_sources: list[T] = dataclasses.field(default_factory=list)
    eval_sources: list[T] = dataclasses.field(default_factory=list)
    max_batch_seconds: float = 400.0  # padded seconds per batch
    pool_size: int = 20000  # clips drawn per sampler chunk
    seed: int = 42

    @classmethod
    def __pre_deserialize__(cls, d: dict[Any, Any]) -> dict[Any, Any]:
        return NamedEntries.to_lists(d, ("train_sources", "eval_sources"))

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        return NamedEntries.to_maps(d, ("train_sources", "eval_sources"))
