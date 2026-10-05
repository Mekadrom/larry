import logging
import sys
from pathlib import Path
from typing import Any

from accelerate import PartialState
from datasets.load import camelcase_to_snakecase

from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.config.training.training_config import TrainingConfig
from larry.common.metrics.metrics import Metrics, WandBMetrics, TensorBoardMetrics, NoneMetrics
from larry.common.training import trainers
from larry.common.training.training_model import LarryModel
from larry.common.utils import load_yaml_config
from larry.common.utils.types import FactoryRegistry

_metrics: FactoryRegistry[Metrics] = {
    "wandb": WandBMetrics,
    "tensorboard": TensorBoardMetrics,
    "tb": TensorBoardMetrics,
}


class Training:
    config: TrainingConfig
    trainer: trainers.TrainerBase

    def __init__(self, config_file: Path, log_dir: Path, resume_dir: str | None = None) -> None:
        self.log = logging.getLogger(type(self).__name__)

        self.log.info(f"Loading pipeline file from yaml config_file={config_file}")
        definition = load_yaml_config(config_file)
        self.log.info("Training config loaded.")

        training_config_dict: dict[str, Any] = definition.get("training", {})
        metrics_backend = training_config_dict.get("metricsBackend", "tensorboard")
        self.log.info(f"Using {metrics_backend} metrics backend.")

        training_config_dict = {camelcase_to_snakecase(k): v for k, v in training_config_dict.items()}

        self.config = TrainingConfig(**training_config_dict)

        trainer_config_dict = definition.get("trainer", {})
        trainer_type_name = trainer_config_dict.get("name")

        model_config_dict: dict[str, Any] = definition.get("model", {})
        model_type_name = model_config_dict.get("name")

        trainer_config = self._build_trainer_config(trainer_config_dict)
        model_config = self._build_model_config(model_config_dict)

        state = PartialState()
        metrics = _metrics[metrics_backend](log_dir) if state.is_main_process else NoneMetrics(log_dir=log_dir)

        model_type = LarryModel.REGISTRY.resolve(model_type_name)
        model = model_type(model_config)

        trainer_type = trainers.TrainerBase.REGISTRY.resolve(trainer_type_name)
        self.trainer = trainer_type(model, trainer_config, metrics, resume_dir=resume_dir)

        self.log.info(f"trainer={self.trainer}")
        self.log.info(f"trainer_config={trainer_config}")

    def _build_trainer_config(self, trainer_config_dict: dict[str, Any]) -> LarryTrainerConfig:
        trainer_config_type_name = trainer_config_dict.get("config")
        trainer_config_type = LarryTrainerConfig.REGISTRY.resolve(trainer_config_type_name)

        self.log.info(f"Using trainer_config_type_name={trainer_config_type_name}")

        trainer_config_overrides = trainer_config_dict.get("overrides", [])
        self.log.info(f"Applying trainer_config_overrides={trainer_config_overrides}")

        trainer_config: LarryTrainerConfig = trainer_config_type(
            **{i["name"]: i["value"] for i in trainer_config_overrides}
        )
        return trainer_config

    def _build_model_config(self, model_config_dict: dict[str, Any]) -> ModelConfig:
        model_config_type_name = model_config_dict.get("config")
        model_config_type = ModelConfig.REGISTRY.resolve(model_config_type_name)

        self.log.info(f"Using model_config_type_name={model_config_type_name}.")

        model_config_overrides = model_config_dict.get("overrides", [])
        self.log.info(f"Applying trainer_config_overrides={model_config_overrides}")

        model_config = model_config_type(
            **{i["name"]: i["value"] for i in model_config_overrides}
        )
        return model_config

    def run(self) -> None:
        """Runs preprocessing."""
        self.log.info("Beginning training...")
        try:
            self.trainer.run()
            self.log.info("Training finished.")
        except:
            self.log.exception("Error during training:")
            sys.exit(-2)
