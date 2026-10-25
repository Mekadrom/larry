from typing import Any

import torch

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.criteria.criterion import LossTerm, SamplableCriterion
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

    def flow_term(self, model_output: O) -> LossTerm:
        # each row's mean squared error counts once, so long and short rows weigh the same
        values = self.sample_losses(model_output, model_output.targets) / self.sample_elements(model_output).clamp(
            min=1)
        return LossTerm(weight=1.0, sums=values, counts=torch.ones_like(values))

    def flow_loss(self, model_output: O) -> torch.Tensor:
        return self.flow_term(model_output).value

    def loss_terms(self, model_output: O, **batch_kwargs: Any) -> dict[str, LossTerm]:
        return {"flow": self.flow_term(model_output)}
