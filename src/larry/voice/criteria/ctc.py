from typing import Any

import torch
import torch.nn.functional as F

from larry.common.criteria.criterion import LossTerm, SamplableCriterion
from larry.voice.config.criterion.voice_criterion_configs import CTCCriterionConfig
from larry.voice.model.voice_models import SpeechEncoderModelOutput


class CTCCriterion(SamplableCriterion[SpeechEncoderModelOutput, CTCCriterionConfig]):
    """An implementation of ``Criterion`` for performing standard CTC loss calculation."""

    def loss_terms(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            model_output: SpeechEncoderModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
            **batch_kwargs: Any,
    ) -> dict[str, LossTerm]:
        """One weighted CTC term per head, each row divided by its target length as ``reduction="mean"`` does."""
        terms = {}
        for name, ctc_log_probs in model_output.ctc_log_probs.items():
            losses = F.ctc_loss(
                ctc_log_probs.transpose(0, 1),
                targets,
                model_output.ctc_lengths,
                target_lengths,
                blank=self.config.blank,
                reduction="none",
                zero_infinity=self.config.zero_infinity,
            )
            values = losses / target_lengths.to(losses.device).clamp(min=1)
            weight = self.config.ctc_layer_loss_weights.get(name, 1.0)
            terms[f"ctc/{name}"] = LossTerm(weight=weight, sums=values, counts=torch.ones_like(values))

        return terms

    def sample_losses(
            self,
            model_output: SpeechEncoderModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Simple forward through the shared ``_ctc_loss`` helper function.

        Does no reduction; for eval.
        """
        return self._ctc_loss(model_output, targets, target_lengths, reduction="none")

    def _ctc_loss(
            self,
            model_output: SpeechEncoderModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
            reduction: str,
    ) -> torch.Tensor:
        """Delegates to the well-known CTC loss function provided by pytorch.

        Provides support for an additional CTC head on the FSQ outputs.
        """
        loss = torch.tensor(0, device=model_output.logits.device)
        for name, ctc_log_probs in model_output.ctc_log_probs.items():
            loss_weight = self.config.ctc_layer_loss_weights.get(name, 1.0)
            loss = loss + (loss_weight * F.ctc_loss(
                ctc_log_probs.transpose(0, 1),
                targets,
                model_output.ctc_lengths,
                target_lengths,
                blank=self.config.blank,
                reduction=reduction,
                zero_infinity=self.config.zero_infinity,
            ))

        return loss
