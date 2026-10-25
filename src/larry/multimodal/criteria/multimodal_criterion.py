from typing import Any

import torch
import torch.nn.functional as F

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.common.metrics.metrics import AggregateMetricEntry, MetricDict
from larry.multimodal.config.criterion.criterion_configs import LarryMultimodalCriterionConfig
from larry.multimodal.model.multimodal_model import LarryMultimodalModelOutput


class LarryMultimodalCriterion(Criterion[LarryMultimodalModelOutput, LarryMultimodalCriterionConfig]):
    def __init__(self, config: LarryMultimodalCriterionConfig) -> None:
        super().__init__(config)
        self.flow = ConditionalFlowMatchingCriterion(CriterionConfig())

    @staticmethod
    def next_in_loss(loss_mask: torch.Tensor) -> torch.Tensor:
        # the loss mask labels target tokens; position t predicts token t + 1
        shifted = torch.zeros_like(loss_mask)
        shifted[:, :-1] = loss_mask[:, 1:]
        return shifted

    def text_loss(self, model_output: LarryMultimodalModelOutput, in_loss: torch.Tensor) -> torch.Tensor | None:
        targets = model_output.text_targets.masked_fill(~in_loss[model_output.text_route], -100)
        if not (targets >= 0).any():
            return None

        return F.cross_entropy(
            model_output.logits.float(), targets, ignore_index=-100, label_smoothing=self.config.text_label_smoothing
        )

    def voice_loss(self, model_output: LarryMultimodalModelOutput, in_loss: torch.Tensor) -> torch.Tensor | None:
        if model_output.voice_logits is None or model_output.voice_targets is None:
            return None

        # (n, k) group targets, already -100 after <eov>; positions out of the loss drop the whole group
        targets = model_output.voice_targets.masked_fill(~in_loss[model_output.voice_route][:, None], -100)
        if not (targets >= 0).any():
            return None

        return F.cross_entropy(
            model_output.voice_logits.flatten(0, 1).float(),
            targets.flatten(),
            ignore_index=-100,
            label_smoothing=self.config.voice_label_smoothing,
        )

    def image_loss(self, model_output: LarryMultimodalModelOutput) -> torch.Tensor | None:
        if model_output.image_flow is None:
            return None

        return self.flow.flow_loss(model_output.image_flow)

    def forward(
            self,
            model_output: LarryMultimodalModelOutput,
            loss_mask: torch.Tensor | None = None,
            **batch_kwargs: Any,
    ) -> torch.Tensor:
        if loss_mask is None:
            raise ValueError("the multimodal criterion needs the collator's loss_mask")

        in_loss = self.next_in_loss(loss_mask.to(model_output.text_route.device))
        losses = {
            "text": (self.config.text_weight, self.text_loss(model_output, in_loss)),
            "voice": (self.config.voice_weight, self.voice_loss(model_output, in_loss)),
            "image": (self.config.image_weight, self.image_loss(model_output)),
        }

        # eval runs without grad; keeps train and eval curves under separate tags
        split = "train" if torch.is_grad_enabled() else "eval"
        metrics = model_output.metrics if model_output.metrics is not None else MetricDict()
        total = model_output.logits.new_zeros(())
        for name, (weight, loss) in losses.items():
            if loss is None:
                continue

            total = total + weight * loss
            metrics.add_metric(AggregateMetricEntry(tag=f"{split}/loss/{name}", total=float(loss.detach()), n_items=1))

        model_output.metrics = metrics
        return total
