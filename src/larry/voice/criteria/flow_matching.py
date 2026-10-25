from typing import Any

import torch

from larry.common.criteria.criterion import LossTerm
from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig
from larry.voice.model.voice_models import SpeechDecoderModelOutput


class SpeechDecoderCriterion(ConditionalFlowMatchingCriterion[SpeechDecoderModelOutput, SpeechDecoderCriterionConfig]):
    def kl_scale(self, step: int | None) -> float:
        if step is None or self.config.kl_warmup_steps <= 0:
            return self.config.kl_weight

        return self.config.kl_weight * min(1.0, step / self.config.kl_warmup_steps)

    def prior_term(self, model_output: SpeechDecoderModelOutput) -> LossTerm:
        mask = model_output.frame_mask.unsqueeze(1)
        squared = (model_output.prior - model_output.prior_targets).square() * mask
        # averaged over every valid frame and mel bin in the batch
        counts = mask.sum(dim=(1, 2)) * model_output.prior.shape[1]
        return LossTerm(weight=self.config.prior_weight, sums=squared.sum(dim=(1, 2)), counts=counts)

    def prior_loss(self, model_output: SpeechDecoderModelOutput) -> torch.Tensor:
        return self.prior_term(model_output).value

    def loss_terms(
            self,
            model_output: SpeechDecoderModelOutput,
            step: int | None = None,
            **batch_kwargs: Any,
    ) -> dict[str, LossTerm]:
        kl = model_output.kl
        return {
            "flow": self.flow_term(model_output),
            "prior": self.prior_term(model_output),
            "kl": LossTerm(weight=self.kl_scale(step), sums=kl, counts=torch.ones_like(kl)),
        }
