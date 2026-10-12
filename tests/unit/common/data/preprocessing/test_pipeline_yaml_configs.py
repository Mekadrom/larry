import dataclasses
import inspect
import os
import re
from pathlib import Path
from typing import Any

import pytest

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig, SingleColumnMapperConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance, ProvenanceAncestry
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.common.utils import load_yaml_config
from larry.common.utils.init_helpers import import_submodules

import_submodules("larry", skip=("larry.common.training", "larry.voice.training", "larry.text.training",
                                 "larry.image.training"))

CONFIG_ROOT = Path(__file__).resolve().parents[5] / "configs" / "preprocessing"
YAML_FILES = sorted(CONFIG_ROOT.rglob("*.yaml"))


def _steps() -> list[tuple[str, int, dict[str, Any]]]:
    steps = []
    for path in YAML_FILES:
        definition = load_yaml_config(path) or {}
        for i, step in enumerate(definition.get("pipeline") or []):
            steps.append((str(path.relative_to(CONFIG_ROOT)), i, step))
    return steps


STEPS = _steps()


def _provenance() -> Provenance:
    return Provenance(
        id_columns=[],
        produced_on="",
        git_commit="",
        git_is_dirty=None,
        ancestry=ProvenanceAncestry(datasets=set(), models=set()),
    )


def _build_config(step: dict[str, Any]) -> PreprocessorConfig:
    config_type = PreprocessorConfig.REGISTRY.resolve(step["config"])
    return config_type(**(step.get("inputs") or {}))


def test_yaml_files_were_found() -> None:
    assert YAML_FILES
    assert STEPS


@pytest.mark.parametrize(("path", "index", "step"), STEPS, ids=[f"{p}[{i}]" for p, i, _ in STEPS])
def test_every_pipeline_step_resolves_and_constructs(
        path: str, index: int, step: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: [])

    preprocessor_type = Preprocessor.REGISTRY.resolve(step["name"])
    config = _build_config(step)

    preprocessor_type(_provenance(), config)


@pytest.mark.parametrize(("path", "index", "step"), STEPS, ids=[f"{p}[{i}]" for p, i, _ in STEPS])
def test_mapper_column_lists_reflect_configured_column_fields(path: str, index: int, step: dict[str, Any]) -> None:
    config = _build_config(step)
    if not isinstance(config, MapperConfig):
        pytest.skip("not a mapper")

    values = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    for name, value in values.items():
        if not isinstance(value, str) or name in ("input_column", "output_column"):
            continue
        if name.endswith("_output_column"):
            assert value in config.output_columns, f"{name}={value!r} missing from {config.output_columns}"
        elif name.endswith("_column"):
            assert value in config.input_columns, f"{name}={value!r} missing from {config.input_columns}"

    if isinstance(config, SingleColumnMapperConfig):
        assert config.input_columns == [config.input_column]
        assert config.output_columns == [config.output_column]


def _config_attributes_read_by(preprocessor_type: type) -> set[str]:
    names = set()
    for cls in preprocessor_type.__mro__:
        if not cls.__module__.startswith("larry."):
            continue
        names.update(re.findall(r"self\.config\.(\w+)", inspect.getsource(cls)))
    return names


@pytest.mark.parametrize(("path", "index", "step"), STEPS, ids=[f"{p}[{i}]" for p, i, _ in STEPS])
def test_preprocessor_only_reads_config_attributes_that_exist(path: str, index: int, step: dict[str, Any]) -> None:
    preprocessor_type = Preprocessor.REGISTRY.resolve(step["name"])
    config = _build_config(step)

    missing = sorted(a for a in _config_attributes_read_by(preprocessor_type) if not hasattr(config, a))

    assert not missing, f"{preprocessor_type.__name__} reads config attributes {type(config).__name__} lacks: {missing}"


@pytest.mark.parametrize(("path", "index", "step"), STEPS, ids=[f"{p}[{i}]" for p, i, _ in STEPS])
def test_mapper_column_lists_have_no_duplicates(path: str, index: int, step: dict[str, Any]) -> None:
    config = _build_config(step)
    if not isinstance(config, MapperConfig):
        pytest.skip("not a mapper")

    assert len(set(config.input_columns)) == len(config.input_columns), config.input_columns
    assert len(set(config.output_columns)) == len(config.output_columns), config.output_columns


def _row_replacing_steps() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    steps = []
    for path in YAML_FILES:
        definition = load_yaml_config(path) or {}
        for step in definition.get("pipeline") or []:
            if "copy_columns" in (step.get("inputs") or {}):
                steps.append((str(path.relative_to(CONFIG_ROOT)), definition, step))
    return steps


ROW_REPLACING_STEPS = _row_replacing_steps()


@pytest.mark.parametrize(("path", "definition", "step"), ROW_REPLACING_STEPS, ids=[p for p, _, _ in ROW_REPLACING_STEPS])
def test_row_replacing_steps_carry_the_provenance_id_columns(
        path: str, definition: dict[str, Any], step: dict[str, Any]
) -> None:
    id_columns = (((definition.get("preprocessor") or {}).get("inputs") or {}).get("provenance") or {}).get("id_columns")

    missing = [c for c in id_columns or [] if c not in step["inputs"]["copy_columns"]]

    assert not missing, f"{step['name']} drops provenance id_columns {missing}"
