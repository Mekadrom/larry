import dataclasses

from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig


@dataclasses.dataclass(kw_only=True)
class SCOTUSSpeechSegmentationAligningMapperConfig(SpeechSegmentationAligningMapperConfig):
    text_column: str = "transcript"
    docket_column: str = "docket"
    date_argued_column: str = "date_argued"
    pdf_bytes_column: str = "pdf_bytes"

    stage_direction_pattern: str | None = r"(?i)\((?:laughter|pause|nods|indicating|inaudible|crosstalk)[^)]*\)"
