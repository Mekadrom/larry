from typing import Any, Literal

import torch
import torch.nn.functional as F

from larry.common.config.criteria.criterion_configs import MelReconstructionCriterionConfig
from larry.common.criteria.criterion import Criterion, SamplableCriterion
from larry.common.model.model import LarrySpeechDecoderModelOutput


class MelReconstructionCriterion(
    SamplableCriterion[LarrySpeechDecoderModelOutput],
    Criterion[LarrySpeechDecoderModelOutput, MelReconstructionCriterionConfig],
):
    def forward(self, model_output: LarrySpeechDecoderModelOutput, **batch_kwargs: Any) -> torch.Tensor:
        return self._reconstruction_loss(model_output, reduction="mean")

    def sample_losses(
            self,
            model_output: LarrySpeechDecoderModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self._reconstruction_loss(model_output, reduction="none")

    def _reconstruction_loss(
            self,
            model_output: LarrySpeechDecoderModelOutput,
            reduction: Literal["none", "mean"],
            **batch_kwargs: Any
    ) -> torch.Tensor:
        # todo: apply masking
        loss = torch.tensor(0, device=model_output.logits.device)
        if self.config.l1_weight > 0.0:
            loss = loss + F.l1_loss(model_output.logits, model_output.target_mels, reduction=reduction)

        if self.config.mse_weight > 0.0:
            loss = loss + F.mse_loss(model_output.logits, model_output.target_mels, reduction=reduction)

        return loss
