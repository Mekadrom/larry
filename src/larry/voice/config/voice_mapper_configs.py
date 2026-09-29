import dataclasses
from typing import Literal

from larry.common.config.mapper_configs import MapperConfig
from larry.voice.config.voice_configs import VoiceConfig


@dataclasses.dataclass(kw_only=True)
class VoiceMapperConfig(VoiceConfig, MapperConfig):
    """Super class for all DTOs related to voice dataset preprocessing."""

    input_column: str = "audio"
    output_column: str | None = "audio"


@dataclasses.dataclass(kw_only=True)
class AudioColumnCastMapperConfig(VoiceMapperConfig):
    decode: bool = False


@dataclasses.dataclass(kw_only=True)
class MelExtractingMapperConfig(VoiceMapperConfig):
    n_fft: int = 1024
    hop_length: int = 256
    f_min: float = 0.0
    f_max: float | None = None
    n_mels: int = 80
    power: int = 2

    audio_dtype: Literal["float16", "float32"] = "float16"

    output_column: str | None = "mels"
