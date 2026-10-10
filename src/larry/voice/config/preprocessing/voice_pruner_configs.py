import dataclasses

from larry.common.config.data.preprocessing.pruner_configs import PrunerConfig
from larry.voice.config.preprocessing.voice_configs import VoiceConfig


@dataclasses.dataclass(kw_only=True)
class VoicePrunerConfig(VoiceConfig, PrunerConfig):
    input_columns: list[str] = dataclasses.field(default_factory=lambda: ["audio"])


@dataclasses.dataclass(kw_only=True)
class AudioDurationColumnPrunerConfig(VoicePrunerConfig):
    input_columns: list[str] = dataclasses.field(default_factory=lambda: ["duration_s"])
    max_duration: float = 20.0
    min_duration: float = 0.1


@dataclasses.dataclass(kw_only=True)
class AudioQualityPrunerConfig(VoicePrunerConfig):
    min_rms_dbfs: float
    min_active_speech_ratio: float
    min_snr: float
    max_clipping_portion: float


@dataclasses.dataclass(kw_only=True)
class AudioQualityPrunerConfigForASR(AudioQualityPrunerConfig):
    min_rms_dbfs: float = -45.0
    min_active_speech_ratio: float = 0.3
    min_snr: float = 5.0
    max_clipping_portion: float = 0.001


@dataclasses.dataclass(kw_only=True)
class AudioQualityPrunerConfigForTTS(AudioQualityPrunerConfig):
    min_rms_dbfs: float = -40.0
    min_active_speech_ratio: float = 0.5
    min_snr: float = 15.0
    max_clipping_portion: float = 0.0001
