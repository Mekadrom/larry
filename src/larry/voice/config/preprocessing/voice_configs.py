import dataclasses

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig


@dataclasses.dataclass(kw_only=True)
class VoiceConfig(PreprocessorConfig):
    sample_rate: int = 16000
