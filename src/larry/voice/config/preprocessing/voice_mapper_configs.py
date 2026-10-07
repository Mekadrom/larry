import dataclasses
from typing import Literal

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig
from larry.voice.config.preprocessing.voice_configs import VoiceConfig


@dataclasses.dataclass(kw_only=True)
class VoiceMapperConfig(VoiceConfig, SingleColumnMapperConfig):
    input_column: str = "audio"
    output_column: str = "audio"


@dataclasses.dataclass(kw_only=True)
class AudioColumnCastMapperConfig(VoiceMapperConfig):
    decode: bool = False


@dataclasses.dataclass(kw_only=True)
class UrlAudioMapperConfig(UrlMapperConfig, VoiceMapperConfig):
    input_column: str = "url"
    output_column: str = "audio"

    download_cache_dir: str = "/tmp/larry/audio_url_mapper"

    duration_column: str | None = None

    hash_column: str = "audio_url_hash"


@dataclasses.dataclass(kw_only=True)
class AudioDurationMapperConfig(VoiceMapperConfig):
    output_column: str = "duration_s"

    keep_input: bool = True


@dataclasses.dataclass(kw_only=True)
class MelExtractingMapperConfig(VoiceMapperConfig):
    n_fft: int = 1024
    hop_length: int = 256
    f_min: float = 0.0
    f_max: float | None = None
    n_mels: int = 80
    power: int = 2

    audio_dtype: Literal["float16", "float32"] = "float16"

    output_column: str = "mels"
