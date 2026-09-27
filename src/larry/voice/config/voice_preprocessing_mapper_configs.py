import dataclasses
from typing import Literal

from larry.common.config.preprocessing_mapper_configs import PreprocessingMapperConfig, PruningPreprocessingMapperConfig


@dataclasses.dataclass(kw_only=True)
class VoicePreprocessingMapperConfig(PreprocessingMapperConfig):
    """Super class for all DTOs related to voice dataset preprocessing."""

    sample_rate: int = 16000
    input_column: str = "audio"
    output_column: str | list[dict[str, str]] | None = "audio"


@dataclasses.dataclass(kw_only=True)
class AudioDurationColumnPruningPreprocessingMapperConfig(
    VoicePreprocessingMapperConfig, PruningPreprocessingMapperConfig
):
    max_duration: float = 20.0
    min_duration: float = 0.1


@dataclasses.dataclass(kw_only=True)
class AudioColumnCastPreprocessingMapperConfig(VoicePreprocessingMapperConfig):
    decode: bool = False


@dataclasses.dataclass(kw_only=True)
class AudioQualityPruningPreprocessingMapperConfig(VoicePreprocessingMapperConfig, PruningPreprocessingMapperConfig):
    min_rms_dbfs: float
    min_active_speech_ratio: float
    min_snr: float
    max_clipping_portion: float


@dataclasses.dataclass(kw_only=True)
class AudioQualityPruningPreprocessingMapperConfigForASR(AudioQualityPruningPreprocessingMapperConfig):
    min_rms_dbfs: float = -45.0
    min_active_speech_ratio: float = 0.3
    min_snr: float = 5.0
    max_clipping_portion: float = 0.001


@dataclasses.dataclass(kw_only=True)
class AudioQualityPruningPreprocessingMapperConfigForTTS(AudioQualityPruningPreprocessingMapperConfig):
    min_rms_dbfs: float = -40.0
    min_active_speech_ratio: float = 0.5
    min_snr: float = 15.0
    max_clipping_portion: float = 0.0001


@dataclasses.dataclass(kw_only=True)
class MelExtractingPreprocessingMapperConfig(VoicePreprocessingMapperConfig):
    n_fft: int = 1024
    hop_length: int = 256
    f_min: float = 0.0
    f_max: float | None = None
    n_mels: int = 80
    power: int = 2

    audio_dtype: Literal["float16", "float32"] = "float16"

    output_column: str | list[dict[str, str]] | None = "mels"
