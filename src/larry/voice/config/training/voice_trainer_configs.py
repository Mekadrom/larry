import dataclasses

from larry.common.config.training.trainer_configs import LarryTrainerConfig


@dataclasses.dataclass(kw_only=True)
class VoiceTrainerConfig(LarryTrainerConfig):
    ...


@dataclasses.dataclass(kw_only=True)
class SpeechTokenizerConfig(VoiceTrainerConfig):
    ...
