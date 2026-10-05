from abc import ABC

from larry.common.training.training_model import LarryModel
from larry.voice.config.model.voice_model_configs import VoiceModelConfig


class VoiceModel[C: VoiceModelConfig = VoiceModelConfig](LarryModel[C], ABC):
    ...
