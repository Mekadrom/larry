from typing import Any

import torch

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.criteria.criterion import SamplableCriterion
from larry.common.model import flow


class ConditionalFlowMatchingCriterion(SamplableCriterion[flow.LarryFlowMatchingOutput, CriterionConfig]):
    def sample_losses(
            self,
            model_output: flow.LarryFlowMatchingOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        return ((model_output.logits - targets).square() * model_output.mask.unsqueeze(1)).sum(dim=(1, 2))

    def forward(self, model_output: flow.LarryFlowMatchingOutput, **batch_kwargs: Any) -> torch.Tensor:
        elements = model_output.mask.sum(dim=1) * model_output.targets.shape[1]
        return (self.sample_losses(model_output, model_output.targets) / elements.clamp(min=1)).mean()
