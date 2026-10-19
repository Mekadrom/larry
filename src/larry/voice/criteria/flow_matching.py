from typing import Any

import torch

from larry.common.criteria.criterion import SamplableCriterion
from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig
from larry.voice.model.voice_models import SpeechDecoderModelOutput


class SpeechDecoderCriterion(SamplableCriterion[SpeechDecoderModelOutput, SpeechDecoderCriterionConfig]):
    def sample_losses(
            self,
            model_output: SpeechDecoderModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor
    ) -> torch.Tensor:
        ...

    def forward(self, model_output: SpeechDecoderModelOutput, **batch_kwargs: Any) -> torch.Tensor:
        ...
