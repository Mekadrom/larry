import dataclasses

import torch

from larry.common.metrics.metrics import MetricDict


@dataclasses.dataclass(kw_only=True)
class LarryModelOutput:
    """Superclass for model outputs; most or all models output logits and/or metrics of some kind."""

    logits: torch.Tensor
    metrics: MetricDict | None = None


@dataclasses.dataclass(kw_only=True)
class LarrySpeechTokenizerModelOutput(LarryModelOutput):
    """``SpeechTokenizerModel`` returns CTC log probs, frame lengths, and token_ids in addition to logits or metrics."""

    ctc_log_probs: torch.Tensor
    frame_lengths: torch.Tensor
    token_ids: torch.Tensor
