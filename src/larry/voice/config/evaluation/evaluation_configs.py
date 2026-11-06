import dataclasses

from larry.common.config.config import Config
from larry.voice.config.evaluation.recognizer_configs import SpeechRecognizerConfig


@dataclasses.dataclass(kw_only=True)
class IntelligibilityConfig(Config):
    recognizers: list[SpeechRecognizerConfig] = dataclasses.field(default_factory=list)
    clips_per_source: int = 32
    batch_size: int = 16
    transcript_column: str = "text_normalized"


@dataclasses.dataclass(kw_only=True)
class PretrainedVocoderConfig(Config):
    model_path: str
    revision: str | None = None
    steps: int = 32
    sway_coefficient: float | None = -1.0
    seed: int = 0
