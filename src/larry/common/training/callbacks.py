import dataclasses
import json
import logging

import torch

from larry.common.metrics.metrics import MetricDict
from larry.common.training import trainers


class TrainingCallback:
    """Superclass for all callbacks for doing stuff before and after a given training step."""

    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        """Prior to the step starting, do stuff."""
        return None

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """After the optimizer step, do stuff."""
        return None


class ModelInitLoggingCallback(TrainingCallback):
    """Implementation of ``TrainingCallback`` that logs metrics once at training start (step 0)."""

    def __init__(self) -> None:
        self.log = logging.getLogger(type(self).__name__)

    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        """Logs some model information and run information at the beginning of training."""
        if step == 0:
            self.log.info(f"model: {trainer.model}")
            trainer.metrics.add_text("model/architecture", trainer.accelerator.unwrap_model(trainer.model), step)
            self.log.info(f"model_config: {trainer.model_config}")
            trainer.metrics.add_text("model/model_config", trainer.model_config, step)
            self.log.info(f"n_parameters: {trainer.n_parameters:,}")
            trainer.metrics.add_text("model/parameter_count", f"{trainer.n_parameters:,}", step)
            trainer.metrics.add_text("run/git_commit_hash", trainer.git_commit_hash, step)
            self.log.info(f"trainer_config: {trainer.config}")
            trainer.metrics.add_text(
                "run/trainer_config",
                json.dumps(dataclasses.asdict(trainer.config), default=str, indent="\t"),
                step
            )


class LoggingTrainingCallback(TrainingCallback):
    """Helper/wrapper that delegates to subclass implementation every ``logging_steps`` steps."""

    def __init__(self, logging_steps: int) -> None:
        """Initializes this instance of ``LoggingTrainingCallback``."""
        self.logging_steps = logging_steps

    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        """Delegates every ``self.logging_steps`` steps."""
        if step % self.logging_steps == 0:
            return self._on_step_begin(trainer, step)

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """Delegates every ``self.logging_steps`` steps."""
        if step % self.logging_steps == 0:
            self._on_step_end(trainer, train_loss, step)

    def _on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        """Subclasses override as they see fit."""
        pass

    def _on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """Subclasses override as they see fit."""
        pass
