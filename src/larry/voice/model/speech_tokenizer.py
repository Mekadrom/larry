import torch

from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.model.voice_models import VoiceModel


class SpeechTokenizerModel(VoiceModel[SpeechTokenizerModelConfig]):
    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        ...
