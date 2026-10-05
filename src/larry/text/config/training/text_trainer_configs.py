import dataclasses
from abc import ABC

from larry.common.config.training.trainer_configs import LarryTrainerConfig


@dataclasses.dataclass(kw_only=True)
class TextTrainerConfig(LarryTrainerConfig, ABC):
    ...


@dataclasses.dataclass(kw_only=True)
class SmokeTestLLMTrainerConfig(TextTrainerConfig):
    ...
