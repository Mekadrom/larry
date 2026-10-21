from typing import Any

import torch

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.criteria.criterion import SamplableCriterion
from larry.common.model import flow


class ConditionalFlowMatchingCriterion[
    O: flow.LarryFlowMatchingOutput = flow.LarryFlowMatchingOutput,
    C: CriterionConfig = CriterionConfig,
](SamplableCriterion[O, C]):
    def sample_losses(
            self,
            model_output: O,
            targets: torch.Tensor,
            target_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        return ((model_output.logits - targets).square() * model_output.mask.unsqueeze(1)).sum(dim=(1, 2))

    def sample_elements(self, model_output: O) -> torch.Tensor:
        return model_output.mask.sum(dim=1) * model_output.targets.shape[1]

    def flow_loss(self, model_output: O) -> torch.Tensor:
        losses = self.sample_losses(model_output, model_output.targets)
        return (losses / self.sample_elements(model_output).clamp(min=1)).mean()

    def forward(self, model_output: O, **batch_kwargs: Any) -> torch.Tensor:
        return self.flow_loss(model_output)
