
import dataclasses

import torch

from larry.common.metrics.metrics import MetricDict


@dataclasses.dataclass(kw_only=True)
class LarryModelOutput:
    logits: torch.Tensor
    metrics: MetricDict | None = None
