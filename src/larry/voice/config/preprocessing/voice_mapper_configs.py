import dataclasses
from typing import Literal

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig
from larry.voice.config.preprocessing.voice_configs import VoiceConfig


@dataclasses.dataclass(kw_only=True)
class VoiceSingleColumnMapperConfig(VoiceConfig, SingleColumnMapperConfig):
    input_column: str = "audio"
    output_column: str = "audio"


@dataclasses.dataclass(kw_only=True)
class AudioColumnCastMapperConfig(VoiceSingleColumnMapperConfig):
    decode: bool = False


@dataclasses.dataclass(kw_only=True)
class MelExtractingMapperConfig(VoiceSingleColumnMapperConfig):
    n_fft: int = 1024
    hop_length: int = 256
    f_min: float = 0.0
    f_max: float | None = None
    n_mels: int = 80
    power: int = 2

    audio_dtype: Literal["float16", "float32"] = "float16"

    output_column: str = "mels"
