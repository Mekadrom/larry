import dataclasses

from larry.common.config.config import Config
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class SampleRateConfig(Config):
    """Provides a config with a sample rate."""

    sample_rate: int = 16000


@dataclasses.dataclass(kw_only=True)
class VoiceConfig(PreprocessorConfig, SampleRateConfig):
    """Superclass for all voice preprocessor configs."""


@dataclasses.dataclass(kw_only=True)
class AlignerConfig(SampleRateConfig):
    """Helper dataclass for configuring an aligner that wraps an alignment model."""

    stride: int = 320
    model_id: str = "facebook/wav2vec2-large-960h-lv60-self"
    telomere_size: int = 400
    chunk_seconds: float = 30.0
    overlap_seconds: float = 3.0
    emission_batch_size: int = 8
    wildcard_penalty: float = 1.0


@dataclasses.dataclass(kw_only=True)
class TranscriptConfig(Config):
    """Configures a transcript parser."""


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptConfig(TranscriptConfig):
    """Configures the SCOTUSTranscript parser class specifically."""

    max_note_lines: int = 4
