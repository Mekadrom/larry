import dataclasses

import torch

from larry.common.metrics.metrics import MetricDict


@dataclasses.dataclass(kw_only=True)
class LarryModelOutput:
    logits: torch.Tensor
    metrics: MetricDict | None = None


@dataclasses.dataclass(kw_only=True)
class LarrySpeechTokenizerModelOutput(LarryModelOutput):
    ctc_log_probs: torch.Tensor
    frame_lengths: torch.Tensor
    token_ids: torch.Tensor
