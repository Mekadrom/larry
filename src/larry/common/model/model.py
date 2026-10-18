import dataclasses
from collections.abc import Mapping

import torch

from larry.common.metrics.metrics import MetricDict


@dataclasses.dataclass(kw_only=True)
class LarryModelOutput:
    """Superclass for model outputs; most or all models output logits and/or metrics of some kind."""

    logits: torch.Tensor
    metrics: MetricDict | None = None


@dataclasses.dataclass(kw_only=True)
class LarrySpeechEncoderModelOutput(LarryModelOutput):
    """``SpeechTokenizerModel`` returns CTC log probs, frame lengths, and unit_ids in addition to logits or metrics."""

    ctc_log_probs: Mapping[str, torch.Tensor]
    frame_lengths: torch.Tensor
    unit_ids: torch.Tensor


@dataclasses.dataclass(kw_only=True)
class LarrySpeechDecoderModelOutput(LarryModelOutput):
    speaker_mu: torch.Tensor
    speaker_logvar: torch.Tensor

    target_mels: torch.Tensor
    target_mel_lengths: torch.Tensor
