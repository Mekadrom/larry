import dataclasses
from typing import Literal

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig, \
    MapperConfig
from larry.voice.config.preprocessing.voice_configs import VoiceConfig, AlignerConfig


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


@dataclasses.dataclass(kw_only=True)
class AudioDurationMapperConfig(VoiceMapperConfig):
    output_column: str = "duration_s"

    keep_input: bool = True


@dataclasses.dataclass(kw_only=True)
class SpeechSegmentationAligningMapperConfig(MapperConfig):
    text_column: str
    audio_column: str = "audio"

    sample_rate: int = 16000

    stage_direction_pattern: str | None = None

    min_aligned_fraction: float = 0.8
    max_segment_seconds: float = 25.0
    pad_start_seconds: float = 0.1
    pad_end_seconds: float = 0.3
    max_pause_seconds: float | None = 2.0
    max_chars_per_frame: float = 0.8
    min_anchor_run: int = 4
    block_seconds: float = 60.0
    max_block_seconds: float = 240.0
    max_word_seconds: float = 2.5
    slack_seconds: float = 0.5
    reading_margin_seconds: float = 0.3

    copy_columns: list[str] = dataclasses.field(default_factory=list)

    aligner_config: AlignerConfig = dataclasses.field(default_factory=AlignerConfig)


@dataclasses.dataclass(kw_only=True)
class CTCScoreMapperConfig(VoiceMapperConfig):
    audio_column: str = "audio"
    text_column: str = "text_normalized"
    duration_column: str = "duration_s"

    output_column: str = "ctc_mismatch"
    text_output_column: str | None = "text_normalized"

    max_batch_seconds: float = 400.0

    aligner_config: AlignerConfig = dataclasses.field(default_factory=AlignerConfig)


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
