import dataclasses

from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig


@dataclasses.dataclass(kw_only=True)
class SCOTUSSpeechSegmentationAligningMapperConfig(SpeechSegmentationAligningMapperConfig):
    """Configures a ``SCOTUSSpeechSegmentationAligningMapper``."""

    text_column: str = "transcript"
    pdf_bytes_column: str = "pdf_bytes"
    date_argued_column: str = "date_argued"
    docket_column: str = "docket"

    def required_input_columns(self) -> list[str]:
        """Sets the required input columns to the conjunct of the superclass' and those required for SCOTUS."""
        return super().required_input_columns() + [
            self.pdf_bytes_column, self.date_argued_column, self.docket_column
        ]

    stage_direction_pattern: str | None = r"(?i)\((?:laughter|pause|nods|indicating|inaudible|crosstalk)[^)]*\)"
