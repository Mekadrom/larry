import dataclasses

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig, SingleColumnMapperConfig, UrlMapperConfig
from larry.voice.config.preprocessing.voice_configs import AlignerConfig, SampleRateConfig


@dataclasses.dataclass(kw_only=True)
class UrlAudioMapperConfig(UrlMapperConfig):
    """Config for ``UrlAudioMapper``, setting sensible defaults for config fields from ``UrlMapperConfig``."""

    input_column: str = "url"
    output_column: str = "audio"

    download_cache_dir: str = "/tmp/larry/audio_url_mapper"


@dataclasses.dataclass(kw_only=True)
class AudioDurationMapperConfig(SingleColumnMapperConfig):
    """Config for ``AudioDurationMapper``, setting sensible defaults for the input and output column names."""

    input_column: str = "audio"
    output_column: str = "duration_s"


@dataclasses.dataclass(kw_only=True)
class SpeechSegmentationAligningMapperConfig(SampleRateConfig, MapperConfig):
    """Config for ``SpeechSegmentationAligningMapper``, overriding the input/output columns."""

    audio_column: str = "audio"
    text_column: str = "text_original"

    audio_output_column: str = "audio"
    text_output_column: str = "text_original"
    text_normalized_output_column: str = "text_normalized"
    audio_duration_output_column: str = "duration_s"

    def required_input_columns(self) -> list[str]:
        """Requires audio, text, and any other columns intended to be copied over as input columns."""
        return [self.audio_column, self.text_column, *self.copy_columns]

    def required_output_columns(self) -> list[str]:
        """Sets only the known output columns as required."""
        return [
            "start_s",
            "end_s",
            "speaker_id",
            "avg_word_score",
        ] + [
            self.audio_output_column,
            self.text_output_column,
            self.text_normalized_output_column,
            self.audio_duration_output_column
        ]

    stage_direction_pattern: str | None = None

    min_aligned_fraction: float = 0.8
    max_segment_seconds: float = 25.0
    pad_start_seconds: float = 0.1
    pad_end_seconds: float = 0.3
    max_pause_seconds: float | None = 2.0
    max_chars_per_frame: float = 0.8
    min_anchor_run: int = 4
    detect_unspoken: bool = True
    min_unspoken_anchor_run: int = 8
    min_unspoken_support_run: int = 2
    max_unspoken_words: int = 150
    block_seconds: float = 20.0
    max_block_seconds: float = 240.0
    max_word_seconds: float = 2.5
    slack_seconds: float = 0.5
    reading_margin_seconds: float = 0.3
    use_wildcards: bool = True
    bad_segment_score: float = -3.0
    edge_word_min_score: float = -2.0
    crosstalk_min_turns: int = 4
    crosstalk_max_turn_seconds: float = 2.0
    crosstalk_max_seconds: float = 10.0

    copy_columns: list[str] = dataclasses.field(default_factory=list)

    aligner_config: AlignerConfig = dataclasses.field(default_factory=AlignerConfig)


@dataclasses.dataclass(kw_only=True)
class CTCScoreMapperConfig(SampleRateConfig, MapperConfig):
    """Config for ``CTCScoreMapper``, overriding the required input/output columns."""

    audio_column: str = "audio"
    text_column: str = "text_normalized"
    duration_column: str = "duration_s"

    ctc_mismatch_output_column: str = "ctc_mismatch"
    text_output_column: str | None = "text_normalized"

    def required_input_columns(self) -> list[str]:
        """Requires the audio, text, and duration column for mapping."""
        return [self.audio_column, self.text_column, self.duration_column]

    def required_output_columns(self) -> list[str]:
        """Produces primarily the ctc_mismatch score column and optionally the text_normalized column by default."""
        if self.text_output_column:
            return [self.ctc_mismatch_output_column, self.text_output_column]
        return [self.ctc_mismatch_output_column]

    max_batch_seconds: float = 200.0

    aligner_config: AlignerConfig = dataclasses.field(default_factory=AlignerConfig)


@dataclasses.dataclass(kw_only=True)
class LarrySpeechEncoderUsingMapperConfig(SampleRateConfig, MapperConfig):
    speech_encoder_checkpoint_path: str


@dataclasses.dataclass(kw_only=True)
class LarrySpeechEncoderUnitIdExtractingMapperConfig(LarrySpeechEncoderUsingMapperConfig):
    audio_column: str = "audio"
    unit_ids_output_column: str = "unit_ids"
    frame_lengths_output_column: str = "frame_lengths"

    def required_input_columns(self) -> list[str]:
        return [self.audio_column]

    def required_output_columns(self) -> list[str]:
        return [self.unit_ids_output_column, self.frame_lengths_output_column]

    batch_size: int = 64
    writer_batch_size: int = 64
    num_proc: int = 4


@dataclasses.dataclass(kw_only=True)
class LarrySpeechEncoderASRMapperConfig(SingleColumnMapperConfig, LarrySpeechEncoderUsingMapperConfig):
    text_tokenizer_path: str
    
    input_column: str = "audio"
    output_column: str = "text_normalized"
