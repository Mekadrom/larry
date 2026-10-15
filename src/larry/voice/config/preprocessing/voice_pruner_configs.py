import dataclasses

from larry.common.config.data.preprocessing.pruner_configs import PrunerConfig
from larry.voice.config.preprocessing.voice_configs import VoiceConfig


@dataclasses.dataclass(kw_only=True)
class VoicePrunerConfig(VoiceConfig, PrunerConfig):
    """Superclass for voice pruner configs."""
    input_columns: list[str] = dataclasses.field(default_factory=lambda: ["audio"])


@dataclasses.dataclass(kw_only=True)
class AudioDurationColumnPrunerConfig(VoicePrunerConfig):
    """Config for ``AudioDurationColumnPruner`` to set the max and min audio duration and input column."""
    input_columns: list[str] = dataclasses.field(default_factory=lambda: ["duration_s"])
    max_duration: float = 25.0
    min_duration: float = 0.1


@dataclasses.dataclass(kw_only=True)
class AudioQualityPrunerConfig(VoicePrunerConfig):
    """Config for ``AudioQualityPruner`` to set the acceptable levels of noise or poor audio quality for filtering."""
    min_rms_dbfs: float
    min_active_speech_ratio: float
    min_snr: float
    max_clipping_portion: float


@dataclasses.dataclass(kw_only=True)
class AudioQualityPrunerConfigForASR(AudioQualityPrunerConfig):
    """Config for ``AudioQualityPruner`` instances that sets the quality bar lower for datasets for ASR training."""
    min_rms_dbfs: float = -45.0
    min_active_speech_ratio: float = 0.3
    min_snr: float = 5.0
    max_clipping_portion: float = 0.001


@dataclasses.dataclass(kw_only=True)
class AudioQualityPrunerConfigForTTS(AudioQualityPrunerConfig):
    """Config for ``AudioQualityPruner`` instances that sets the quality bar higher for datasets for TTS training."""
    min_rms_dbfs: float = -40.0
    min_active_speech_ratio: float = 0.5
    min_snr: float = 15.0
    max_clipping_portion: float = 0.0001
