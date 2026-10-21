from typing import Any

import torch

from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig
from larry.voice.model.voice_models import SpeechDecoderModelOutput


class SpeechDecoderCriterion(ConditionalFlowMatchingCriterion[SpeechDecoderModelOutput, SpeechDecoderCriterionConfig]):
    def kl_scale(self, step: int | None) -> float:
        if step is None or self.config.kl_warmup_steps <= 0:
            return self.config.kl_weight

        return self.config.kl_weight * min(1.0, step / self.config.kl_warmup_steps)

    def prior_loss(self, model_output: SpeechDecoderModelOutput) -> torch.Tensor:
        mask = model_output.frame_mask.unsqueeze(1)
        squared = (model_output.prior - model_output.prior_targets).square() * mask
        return squared.sum() / (mask.sum() * model_output.prior.shape[1]).clamp(min=1)

    def forward(
            self,
            model_output: SpeechDecoderModelOutput,
            step: int | None = None,
            **batch_kwargs: Any,
    ) -> torch.Tensor:
        return (
            self.flow_loss(model_output)
            + self.config.prior_weight * self.prior_loss(model_output)
            + self.kl_scale(step) * model_output.kl.mean()
        )
