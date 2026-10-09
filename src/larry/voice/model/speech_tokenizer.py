import dataclasses

import torch
from torch import nn

from larry.common.config.model.model_configs import LLMBlockModuleConfig
from larry.common.model.llm import LarrySimpleEncoderBlock
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.model.fsq import FSQ
from larry.voice.model.voice_models import VoiceModel


class SpeechTokenizerModel(VoiceModel[SpeechTokenizerModelConfig]):
    def __init__(self, config: SpeechTokenizerModelConfig) -> None:
        super().__init__(config)

        block_fields = {f.name for f in dataclasses.fields(LLMBlockModuleConfig)}
        self.encoder1 = nn.Sequential(*[
            LarrySimpleEncoderBlock(LLMBlockModuleConfig(**{
                k: v
                for k, v in dataclasses.asdict(config).items()
                if k in block_fields
            }))
            for _ in range(config.n_layers)
        ])
        self.fsq = FSQ(self.config.d_model)
        self.encoder2 = nn.Sequential(*[
            LarrySimpleEncoderBlock(LLMBlockModuleConfig(**{
                k: v
                for k, v in dataclasses.asdict(config).items()
                if k in block_fields
            }))
            for _ in range(config.n_layers)
        ])

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        ...
