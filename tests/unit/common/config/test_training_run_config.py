import pytest

from larry.common.config.training.definition import TrainingDefinition, TrainingRunConfig
from larry.common.config.training.training_config import _metrics
from larry.common.training.trainers import TrainerBase
from larry.common.training.training_model import LarryModel
from tests.helpers.config_recipes import TRAINING_RECIPES, load_training


def test_training_keys_are_parsed() -> None:
    config = TrainingRunConfig.from_dict({"metrics_backend": "wandb", "provenance": {"id_columns": ["id"]}})

    assert config.metrics_backend == "wandb"
    assert config.provenance.id_columns == ["id"]


def test_the_default_backend_is_registered() -> None:
    assert TrainingRunConfig().metrics_backend in _metrics


@pytest.mark.parametrize("recipe", list(TRAINING_RECIPES))
def test_every_training_recipe_loads_with_a_registered_backend_and_implementations(recipe: str) -> None:
    definition = load_training(recipe)

    assert definition.training.metrics_backend in _metrics
    assert TrainerBase.REGISTRY.for_config(definition.trainer)
    assert LarryModel.REGISTRY.for_config(definition.model)


@pytest.mark.parametrize("recipe", list(TRAINING_RECIPES))
def test_rendered_training_config_loads_back_identically(recipe: str) -> None:
    definition = load_training(recipe)

    assert TrainingDefinition.from_dict(definition.to_dict()) == definition
