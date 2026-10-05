import logging

import torch

from larry.common.metrics.metrics import MetricDict
from larry.common.training import trainers


class TrainingCallback:
    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        return None

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        return None

    def on_epoch_begin(self, trainer: trainers.TrainerBase, epoch: int) -> MetricDict | None:
        return None

    def on_epoch_end(
            self,
            trainer: trainers.TrainerBase,
            eval_loss: torch.Tensor | None,
            epoch: int
    ) -> MetricDict | None:
        return None


class ModelInitLoggingCallback(TrainingCallback):
    def __init__(self) -> None:
        self.log = logging.getLogger(type(self).__name__)

    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        if step == 0:
            self.log.info(f"model: {trainer.model}")
            trainer.metrics.add_text("model/architecture", trainer.model, step)
            self.log.info(f"model_config: {trainer.model_config}")
            trainer.metrics.add_text("model/model_config", trainer.model_config, step)
            self.log.info(f"n_parameters: {trainer.n_parameters}")
            trainer.metrics.add_text("model/parameter_count", trainer.n_parameters, step)
            trainer.metrics.add_text("run/git_commit_hash", trainer.git_commit_hash, step)
            self.log.info(f"trainer_config: {trainer.config}")
            trainer.metrics.add_text("run/trainer_config", trainer.config, step)


class LoggingTrainingCallback(TrainingCallback):
    def __init__(self, logging_steps: int) -> None:
        self.logging_steps = logging_steps

    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        if step % self.logging_steps == 0:
            return self._on_step_begin(trainer, step)

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        if step % self.logging_steps == 0:
            self._on_step_end(trainer, train_loss, step)

    def _on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        pass

    def _on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        pass
