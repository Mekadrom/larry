import logging
from abc import abstractmethod, ABC

import torch

from larry.common.metrics.metrics import MetricDict
from larry.common.training import trainers, callbacks


class VizCallback(callbacks.TrainingCallback, ABC):
    """Superclass for all visualization training callbacks.

    Does visualization every ``viz_steps`` steps, at the end of those steps (after optimizer backward/step).
    """

    def __init__(self, viz_steps: int) -> None:
        """Common initializer for all subclasses."""
        self.log = logging.getLogger(type(self).__name__)
        self.viz_steps = viz_steps

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """On step end, if this step is a visualization step, performs visualization.

        Doesn't return a MetricDict, for some reason choosing to log metrics to the trainer directly instead of
        collecting them.
        """
        if step % self.viz_steps == 0:
            self.viz(trainer, step)

    @abstractmethod
    def viz(self, trainer: trainers.TrainerBase, step: int) -> None:
        """Entrypoint for visualization."""
        ...
