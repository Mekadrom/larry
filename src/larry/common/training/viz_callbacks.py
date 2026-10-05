import logging
from abc import abstractmethod, ABC

import torch

from larry.common.metrics.metrics import MetricDict
from larry.common.training import trainers, callbacks


class VizCallback(callbacks.TrainingCallback, ABC):
    def __init__(self, viz_steps: int) -> None:
        self.viz_steps = viz_steps
        self.log = logging.getLogger(type(self).__name__)

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        if step % self.viz_steps == 0:
            self.viz(trainer, step)

    @abstractmethod
    def viz(self, trainer: trainers.TrainerBase, step: int) -> None:
        ...
