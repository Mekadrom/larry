import dataclasses

import torch

from larry.common.metrics.metrics import MetricDict


@dataclasses.dataclass(kw_only=True)
class LarryModelOutput:
    """Superclass for model outputs; most or all models output logits and/or metrics of some kind."""

    logits: torch.Tensor
    metrics: MetricDict | None = None
