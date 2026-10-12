import logging
from typing import Any

from larry.common.config.training.trainer_configs import LarryTrainerConfig

log = logging.getLogger(__name__)


def build_trainer_config(trainer_config_dict: dict[str, Any]) -> LarryTrainerConfig:
    trainer_config_type_name = trainer_config_dict.get("config")
    trainer_config_type = LarryTrainerConfig.REGISTRY.resolve(trainer_config_type_name)

    log.info(f"Using trainer_config_type_name={trainer_config_type_name}")

    trainer_config_overrides = trainer_config_dict.get("overrides", [])
    log.info(f"Applying trainer_config_overrides={trainer_config_overrides}")

    trainer_config: LarryTrainerConfig = trainer_config_type(**trainer_config_overrides)
    return trainer_config
