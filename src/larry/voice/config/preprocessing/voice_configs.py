import dataclasses
from typing import ClassVar

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class SampleRateConfig:
    """Provides a config with a sample rate."""
    sample_rate: int = 16000


@dataclasses.dataclass(kw_only=True)
class SpeechSourceConfig:
    """A config dataclass for configuring a single dataset source."""
    name: str
    index_path: str  # huggingface repo or local dir: rows with id, duration_s, text_original, speaker_id, etc.
    source_path: str | None = None
    source_name: str | None = None
    source_revision: str | None = None
    source_cache_dir: str | None = None
    split: str = "train"
    source_split: str = "train"
    audio_column: str = "audio"
    duration_column: str = "duration_s"
    text_column: str = "text_normalized"
    weight: float = 1.0  # target share of audio time, not of rows


@dataclasses.dataclass(kw_only=True)
class SpeechMixtureConfig(SampleRateConfig):
    """A config dataclass for defining the entire training and eval mixture of dataset sources and batch hyperparams."""
    train_sources: list[SpeechSourceConfig] = dataclasses.field(default_factory=list)
    eval_sources: list[SpeechSourceConfig] = dataclasses.field(default_factory=list)
    max_batch_seconds: float = 400.0  # padded seconds per batch
    pool_size: int = 20000  # clips drawn per sampler chunk
    seed: int = 42


@dataclasses.dataclass(kw_only=True)
class VoiceConfig(PreprocessorConfig, SampleRateConfig):
    """Superclass for all voice preprocessor configs."""
    ...


@dataclasses.dataclass(kw_only=True)
class AlignerConfig(Registrable, SampleRateConfig, root=True):
    """Helper dataclass for configuring an aligner that wraps an alignment model."""
    REGISTRY: ClassVar[TypeRegistry[AlignerConfig]]

    stride: int = 320
    model_id: str = "facebook/wav2vec2-large-960h-lv60-self"
    telomere_size: int = 400
    chunk_seconds: float = 30.0
    overlap_seconds: float = 3.0
    emission_batch_size: int = 8


@dataclasses.dataclass(kw_only=True)
class TranscriptConfig:
    """Configures a transcript parser."""
    REGISTRY: ClassVar[TypeRegistry[TranscriptConfig]]
    ...


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptConfig(TranscriptConfig):
    """Configures the SCOTUSTranscript parser class specifically."""
    max_note_lines: int = 4
