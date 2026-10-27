import itertools
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any

import torch

from larry.common.metrics.metrics import MetricDict
from larry.common.training import callbacks, trainers


def pick_rows(offsets: list[int], per_source: int, sources: Collection[int] | None = None) -> list[int]:
    """Picks the same rows from each dataset source (or only the given sources) for visualization."""
    rows: list[int] = []
    for source, (start, end) in enumerate(itertools.pairwise(offsets)):
        if sources is not None and source not in sources:
            continue

        span = end - start
        rows += sorted({start + (j * span) // per_source for j in range(min(per_source, span))})

    return rows


class VizCallback(callbacks.TrainingCallback, ABC):
    """Superclass for all visualization training callbacks.

    Does visualization every ``viz_steps`` steps, at the end of those steps (after optimizer backward/step).
    """

    counts_as_training = False

    def __init__(self, trainer: trainers.TrainerBase[Any, Any, Any], viz_steps: int) -> None:
        """Common initializer for all subclasses."""
        super().__init__(trainer)
        self.viz_steps = viz_steps

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """On step end, if this step is a visualization step, performs visualization.

        Doesn't return a MetricDict, for some reason choosing to log metrics to the trainer directly instead of
        collecting them.
        """
        if step % self.viz_steps == 0:
            self.viz(step)

    @abstractmethod
    def viz(self, step: int) -> None:
        """Entrypoint for visualization."""
        ...
