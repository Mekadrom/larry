from pathlib import Path

import pytest

from larry.common.config.training.training_config import TrainingRunConfig, _metrics
from larry.common.utils import load_yaml_dict

TRAINING_CONFIGS = sorted((Path(__file__).parents[4] / "configs" / "training").rglob("*.yaml"))


def test_training_keys_are_parsed() -> None:
    config = TrainingRunConfig(metrics_backend="wandb", provenance={"id_columns": ["id"]})

    assert config.metrics_backend == "wandb"
    assert config.provenance == {"id_columns": ["id"]}


def test_the_default_backend_is_registered() -> None:
    assert TrainingRunConfig().metrics_backend in _metrics


@pytest.mark.parametrize("config_file", TRAINING_CONFIGS, ids=lambda p: p.name)
def test_every_training_config_names_a_registered_backend(config_file: Path) -> None:
    config = TrainingRunConfig(**load_yaml_dict(config_file).get("training", {}))

    assert config.metrics_backend in _metrics
