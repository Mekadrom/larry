from typing import Any, cast

import torch

from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.training.callbacks import LoggingTrainingCallback


class EmittingCallback(LoggingTrainingCallback):
    def _on_step_begin(self, step: int) -> MetricDict | None:
        return MetricDict({"begin": ScalarMetricEntry(tag="begin", scalar=float(step))})

    def _on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        return MetricDict({"end": ScalarMetricEntry(tag="end", scalar=float(step))})


def test_logging_callbacks_return_their_metrics_on_logging_steps_only() -> None:
    callback = EmittingCallback(cast(Any, None), logging_steps=2)

    assert callback.on_step_begin(2) is not None
    assert callback.on_step_end(torch.tensor(0.0), 2) is not None
    assert callback.on_step_begin(3) is None
    assert callback.on_step_end(torch.tensor(0.0), 3) is None
