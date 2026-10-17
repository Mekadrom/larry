import dataclasses
from collections.abc import Mapping
from typing import Any, Self, cast


@dataclasses.dataclass(kw_only=True)
class DatasetSourceConfig:
    """A config dataclass for configuring a single dataset source."""

    name: str
    index_path: str  # huggingface repo or local dir: rows with id, duration_s, text_original, speaker_id, etc.
    source_path: str | None = None
    source_name: str | None = None
    source_revision: str | None = None
    source_cache_dir: str | None = None
    split: str = "train"
    source_split: str = "train"
    weight: float = 1.0  # target share of audio time, not of rows


@dataclasses.dataclass(kw_only=True)
class DatasetMixtureConfig[T: DatasetSourceConfig = DatasetSourceConfig]:
    """A config dataclass for defining the entire training and eval mixture of dataset sources and batch hyperparams."""

    train_sources: list[T] = dataclasses.field(default_factory=list)
    eval_sources: list[T] = dataclasses.field(default_factory=list)
    max_batch_seconds: float = 400.0  # padded seconds per batch
    pool_size: int = 20000  # clips drawn per sampler chunk
    seed: int = 42

    @classmethod
    def source_config_type(cls) -> type[T]:
        """The concrete source config class this mixture builds its sources as."""
        return cast("type[T]", DatasetSourceConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        """Builds the mixture from a raw yaml dict, constructing its sources as ``source_config_type`` instances."""
        raw = dict(raw)
        source_type = cls.source_config_type()
        return cls(
            train_sources=[source_type(**source) for source in raw.pop("train_sources", [])],
            eval_sources=[source_type(**source) for source in raw.pop("eval_sources", [])],
            **raw,
        )
