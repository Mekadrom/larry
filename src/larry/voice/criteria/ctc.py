from typing import Any

import torch
import torch.nn.functional as F

from larry.common.config.criteria.criterion_configs import CTCCriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.model.model import LarrySpeechTokenizerModelOutput
from larry.voice.model.speech_tokenizer import SpeechEncoderModel


class CTCCriterion(Criterion[CTCCriterionConfig, SpeechEncoderModel]):
    """An implementation of ``Criterion`` for performing standard CTC loss calculation."""

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            model_output: LarrySpeechTokenizerModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
            **batch_kwargs: Any,
    ) -> torch.Tensor:
        """Simple forward through the shared ``_ctc_loss`` helper function.

        Reduces by mean; for training.
        """
        return self._ctc_loss(model_output, targets, target_lengths, reduction="mean")

    def sample_losses(
            self,
            model_output: LarrySpeechTokenizerModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Simple forward through the shared ``_ctc_loss`` helper function.

        Does no reduction; for eval.
        """
        return self._ctc_loss(model_output, targets, target_lengths, reduction="none")

    def _ctc_loss(
            self,
            model_output: LarrySpeechTokenizerModelOutput,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
            reduction: str,
    ) -> torch.Tensor:
        """Delegates to the well-known CTC loss function provided by pytorch."""
        return F.ctc_loss(
            model_output.ctc_log_probs.transpose(0, 1),
            targets,
            model_output.frame_lengths,
            target_lengths,
            blank=self.config.blank,
            reduction=reduction,
            zero_infinity=self.config.zero_infinity,
        )
