import logging
from typing import Any

import torch
import yaml

from larry.common.metrics.metrics import MetricDict
from larry.common.training import trainers


class TrainingCallback:
    """Superclass for all callbacks for doing stuff before and after a given training step."""

    counts_as_training: bool = True

    def __init__(self, trainer: trainers.TrainerBase[Any, Any, Any]) -> None:
        """Simple init."""
        self.log = logging.getLogger(type(self).__name__)
        self.trainer = trainer

    def on_step_begin(self, step: int) -> MetricDict | None:
        """Prior to the step starting, do stuff."""
        return None

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """After the optimizer step, do stuff."""
        return None


class ModelInitLoggingCallback(TrainingCallback):
    """Implementation of ``TrainingCallback`` that logs metrics once at the start of each run, including resumes."""

    def __init__(self, trainer: trainers.TrainerBase[Any, Any, Any]) -> None:
        super().__init__(trainer)
        self.logged = False

    def on_step_begin(self, step: int) -> MetricDict | None:
        """Logs some model information and run information at the beginning of training."""
        if self.logged:
            return None
        self.logged = True

        self.log.info(f"model: {self.trainer.model}")
        self.trainer.metrics.add_text(
            "model/architecture",
            self.trainer.accelerator.unwrap_model(self.trainer.model),
            step
        )
        self.log.info(f"model_config: {self.trainer.model_config}")
        self.trainer.metrics.add_text("model/model_config", self._yaml_block(self.trainer.model_config.to_dict()), step)
        self.log.info(f"n_parameters: {self.trainer.n_parameters:,}")
        self.trainer.metrics.add_text("model/parameter_count", f"{self.trainer.n_parameters:,}", step)
        self.trainer.metrics.add_text("run/git_commit_hash", self.trainer.git_commit_hash, step)
        self.log.info(f"trainer_config: {self.trainer.config}")
        self.trainer.metrics.add_text("run/trainer_config", self._yaml_block(self.trainer.config.to_dict()), step)

        provenance = self.trainer.provenance
        if provenance.config is not None:
            self.trainer.metrics.add_text("run/config", self._yaml_block(provenance.config), step)
        if provenance.cli is not None:
            self.log.info(f"cli: {provenance.cli}")
            self.trainer.metrics.add_text("run/cli", f"```\n{provenance.cli}\n```", step)
        return None

    @staticmethod
    def _yaml_block(config: dict[str, Any]) -> str:
        return f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```"


class LoggingTrainingCallback(TrainingCallback):
    """Helper/wrapper that delegates to subclass implementation every ``logging_steps`` steps."""

    def __init__(self, trainer: trainers.TrainerBase[Any, Any, Any], logging_steps: int) -> None:
        """Initializes this instance of ``LoggingTrainingCallback``."""
        super().__init__(trainer)
        self.logging_steps = logging_steps

    def on_step_begin(self, step: int) -> MetricDict | None:
        """Delegates every ``self.logging_steps`` steps."""
        if step % self.logging_steps == 0:
            return self._on_step_begin(step)
        return None

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """Delegates every ``self.logging_steps`` steps."""
        if step % self.logging_steps == 0:
            return self._on_step_end(train_loss, step)
        return None

    def _on_step_begin(self, step: int) -> MetricDict | None:
        """Subclasses override as they see fit."""
        pass

    def _on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """Subclasses override as they see fit."""
        pass


class TrainLossByTokens(LoggingTrainingCallback):
    def _on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """Doesn't return a metrics dict because it doesn't use training steps as the x axis."""
        self.trainer.metrics.add_scalar("train/loss_by_tokens", train_loss.item(), self.trainer.sync_tokens_seen())
