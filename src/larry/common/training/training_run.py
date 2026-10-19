import logging
import sys
from pathlib import Path

from larry.common.config.training import training_config
from larry.common.config.training.definition import TrainingDefinition, TrainingRunConfig
from larry.common.training import trainers


class TrainingRun:
    """Wraps a trainer with the run config and state."""

    state: training_config.TrainingRunState
    config: TrainingRunConfig
    trainer: trainers.TrainerBase

    def __init__(self, definition: TrainingDefinition, log_dir: Path, resume_dir: str | None = None) -> None:
        """Initializes a training run."""
        self.log = logging.getLogger(type(self).__name__)

        self.state = training_config.TrainingRunState(definition, log_dir, resume_dir)
        self.config = self.state.config
        self.trainer = self.state.trainer

    def run(self) -> None:
        """Runs training."""
        self.log.info("Beginning training run...")
        try:
            self.trainer.run()
            self.log.info("Training run finished.")
        except Exception:
            self.log.exception("Error during training run:")
            sys.exit(-2)
