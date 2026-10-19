import dataclasses
import inspect
import os
import re

import pytest

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig, SingleColumnMapperConfig
from larry.common.config.data.preprocessing.pipeline_config import PipelineDefinition
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing.preprocessors import Preprocessor
from tests.helpers.common import import_larry_submodules, provenance
from tests.helpers.config_recipes import PREPROCESSING_RECIPES, load_pipeline, load_pipelines

import_larry_submodules()

PIPELINES = load_pipelines()
STEPS = [(name, step) for name, definition in PIPELINES.items() for step in definition.pipeline]
parametrize_steps = pytest.mark.parametrize(("recipe", "step"), STEPS, ids=[f"{r}[{s.name}]" for r, s in STEPS])


def test_recipes_were_found() -> None:
    assert PREPROCESSING_RECIPES
    assert STEPS


@pytest.mark.parametrize("recipe", list(PREPROCESSING_RECIPES))
def test_every_recipe_loads_and_resolves_its_dataset_provider(recipe: str) -> None:
    definition = load_pipeline(recipe)

    assert DatasetProvider.REGISTRY.for_config(definition.dataset)
    assert all(step.name for step in definition.pipeline)


@pytest.mark.parametrize("recipe", list(PIPELINES))
def test_rendered_config_loads_back_identically(recipe: str) -> None:
    definition = PIPELINES[recipe]

    assert PipelineDefinition.from_dict(definition.to_dict()) == definition


@parametrize_steps
def test_every_pipeline_step_resolves_and_constructs(
        recipe: str, step: PreprocessorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: [])

    Preprocessor.REGISTRY.for_config(step)(provenance(), step)


@parametrize_steps
def test_mapper_column_lists_reflect_configured_column_fields(recipe: str, step: PreprocessorConfig) -> None:
    if not isinstance(step, MapperConfig):
        pytest.skip("not a mapper")

    values = {f.name: getattr(step, f.name) for f in dataclasses.fields(step)}
    for name, value in values.items():
        if not isinstance(value, str) or name in ("input_column", "output_column", "name"):
            continue
        if name.endswith("_output_column"):
            assert value in step.output_columns, f"{name}={value!r} missing from {step.output_columns}"
        elif name.endswith("_column"):
            assert value in step.input_columns, f"{name}={value!r} missing from {step.input_columns}"

    if isinstance(step, SingleColumnMapperConfig):
        assert step.input_columns == [step.input_column]
        assert step.output_columns == [step.output_column]


def _config_attributes_read_by(preprocessor_type: type) -> set[str]:
    names = set()
    for cls in preprocessor_type.__mro__:
        if not cls.__module__.startswith("larry."):
            continue
        names.update(re.findall(r"self\.config\.(\w+)", inspect.getsource(cls)))
    return names


@parametrize_steps
def test_preprocessor_only_reads_config_attributes_that_exist(recipe: str, step: PreprocessorConfig) -> None:
    preprocessor_type = Preprocessor.REGISTRY.for_config(step)

    missing = sorted(a for a in _config_attributes_read_by(preprocessor_type) if not hasattr(step, a))

    assert not missing, f"{preprocessor_type.__name__} reads config attributes {type(step).__name__} lacks: {missing}"


@parametrize_steps
def test_mapper_column_lists_have_no_duplicates(recipe: str, step: PreprocessorConfig) -> None:
    if not isinstance(step, MapperConfig):
        pytest.skip("not a mapper")

    assert len(set(step.input_columns)) == len(step.input_columns), step.input_columns
    assert len(set(step.output_columns)) == len(step.output_columns), step.output_columns


ROW_REPLACING_STEPS = [(name, step) for name, step in STEPS if hasattr(step, "copy_columns")]


@pytest.mark.parametrize(("recipe", "step"), ROW_REPLACING_STEPS, ids=[r for r, _ in ROW_REPLACING_STEPS])
def test_row_replacing_steps_carry_the_provenance_id_columns(recipe: str, step: PreprocessorConfig) -> None:
    id_columns = PIPELINES[recipe].preprocessor.provenance.id_columns

    missing = [c for c in id_columns if c not in step.copy_columns]

    assert not missing, f"{step.name} drops provenance id_columns {missing}"
