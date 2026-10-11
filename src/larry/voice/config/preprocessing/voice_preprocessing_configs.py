import dataclasses
from typing import Literal, Any

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class SpeakerValidationSplittingPreprocessorConfig(PreprocessorConfig):
    duration_column: str = "duration_s"
    speaker_id_column: str = "speaker_id"

    mode: Literal["matching", "random"] = "random"
    matching_values: list[Any] | None = None
    validation_percent: float | None = None
    operate_on_split: str | None = None
