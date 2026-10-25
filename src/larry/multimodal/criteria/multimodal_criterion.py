from typing import Any

import torch
import torch.nn.functional as F

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.criteria.criterion import LossTerm, TermCriterion
from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.multimodal.config.criterion.criterion_configs import LarryMultimodalCriterionConfig
from larry.multimodal.model.multimodal_model import LarryMultimodalModelOutput


class LarryMultimodalCriterion(TermCriterion[LarryMultimodalModelOutput, LarryMultimodalCriterionConfig]):
    def __init__(self, config: LarryMultimodalCriterionConfig) -> None:
        super().__init__(config)
        self.flow = ConditionalFlowMatchingCriterion(CriterionConfig())

    @staticmethod
    def next_in_loss(loss_mask: torch.Tensor) -> torch.Tensor:
        # the loss mask labels target tokens; position t predicts token t + 1
        shifted = torch.zeros_like(loss_mask)
        shifted[:, :-1] = loss_mask[:, 1:]
        return shifted

    @staticmethod
    def per_row(rows: torch.Tensor, values: torch.Tensor, n_rows: int) -> torch.Tensor:
        # routed positions are flattened row-major; fold their values back into one total per batch row
        return values.new_zeros(n_rows).index_add_(0, rows, values)

    def token_term(
            self,
            weight: float,
            logits: torch.Tensor | None,
            targets: torch.Tensor | None,
            route: torch.Tensor,
            in_loss: torch.Tensor,
            label_smoothing: float,
    ) -> LossTerm:
        n_rows = route.shape[0]
        if logits is None or targets is None:
            zeros = torch.zeros(n_rows, device=route.device)
            return LossTerm(weight=weight, sums=zeros, counts=zeros.clone())

        # (n,) or (n, k) targets; a position out of the loss drops every target it predicts
        position_in_loss = in_loss[route].view(-1, *([1] * (targets.dim() - 1)))
        targets = targets.masked_fill(~position_in_loss, -100)
        losses = F.cross_entropy(
            logits.flatten(0, targets.dim() - 1).float(),
            targets.flatten(),
            ignore_index=-100,
            reduction="none",
            label_smoothing=label_smoothing,
        ).view(targets.shape)

        valid = (targets >= 0).float()
        if targets.dim() > 1:
            losses, valid = losses.sum(dim=1), valid.sum(dim=1)

        rows = route.nonzero()[:, 0]

        # averaged over every trained token in the batch
        return LossTerm(weight=weight, sums=self.per_row(rows, losses, n_rows),
                        counts=self.per_row(rows, valid, n_rows))

    def text_term(self, model_output: LarryMultimodalModelOutput, in_loss: torch.Tensor) -> LossTerm:
        return self.token_term(
            self.config.text_weight, model_output.logits, model_output.text_targets, model_output.text_route, in_loss,
            self.config.text_label_smoothing,
        )

    def voice_term(self, model_output: LarryMultimodalModelOutput, in_loss: torch.Tensor) -> LossTerm:
        return self.token_term(
            self.config.voice_weight, model_output.voice_logits, model_output.voice_targets, model_output.voice_route,
            in_loss, self.config.voice_label_smoothing,
        )

    def image_term(self, model_output: LarryMultimodalModelOutput) -> LossTerm:
        n_rows = model_output.image_route.shape[0]
        sums = torch.zeros(n_rows, device=model_output.image_route.device)
        counts = sums.clone()
        if model_output.image_flow is not None:
            # one flow value per generated image, in the order of the rows that hold one
            rows = model_output.image_route.any(dim=1).nonzero()[:, 0]
            term = self.flow.flow_term(model_output.image_flow)
            sums = self.per_row(rows, term.sums.float(), n_rows)
            counts = self.per_row(rows, term.counts.float(), n_rows)

        return LossTerm(weight=self.config.image_weight, sums=sums, counts=counts)

    def loss_terms(
            self,
            model_output: LarryMultimodalModelOutput,
            loss_mask: torch.Tensor | None = None,
            **batch_kwargs: Any,
    ) -> dict[str, LossTerm]:
        if loss_mask is None:
            raise ValueError("the multimodal criterion needs the collator's loss_mask")

        in_loss = self.next_in_loss(loss_mask.to(model_output.text_route.device))
        return {
            "text": self.text_term(model_output, in_loss),
            "voice": self.voice_term(model_output, in_loss),
            "image": self.image_term(model_output),
        }
