import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from larry.common.config.config import ConfigValues, RunRecord
from larry.common.config.data.preprocessing.pipeline_config import PipelineDefinition
from larry.common.data.preprocessing.pipeline import Pipeline
from tests.helpers.common import import_larry_submodules, write_parquet_splits

import_larry_submodules()


def _pipeline(tmp_path: Path, source: Path, split: str | None, fractions: dict[str, float] | None) -> Pipeline:
    definition: dict[str, Any] = {
        "preprocessor": {"default_batch_size": 2, "provenance": {"id_columns": ["text"]}},
        "dataset": {
            "type": "DefaultDatasetConfig",
            "path": "parquet",
            "data_dir": str(source),
            "split": split,
            "cache_dir": str(tmp_path / "cache"),
        },
        "pipeline": {
            "rename_text": {
                "type": "ColumnRenamingPreprocessorConfig",
                "column_name_mappings": {"text": "text_original"},
            },
        },
    }
    if fractions is not None:
        definition["splits"] = fractions
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(yaml.safe_dump(definition, sort_keys=False))
    return Pipeline(ConfigValues([config_file]).load(PipelineDefinition), tmp_path / "out")


def _saved(out: Path) -> dict[str, list[str]]:
    return {
        d.name: sorted(pq.read_table(sorted(d.glob("*.parquet")))["text_original"].to_pylist())
        for d in sorted(out.iterdir())
        if d.is_dir()
    }


SPLITS = {"train": ["a", "b", "c", "d"], "validation": ["e"], "test": ["f", "g"]}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return write_parquet_splits(tmp_path / "source", {name: {"text": texts} for name, texts in SPLITS.items()})


def test_input_splits_keep_their_names(tmp_path: Path, source: Path) -> None:
    _pipeline(tmp_path, source, split=None, fractions=None).preprocess_all()

    assert _saved(tmp_path / "out") == SPLITS


def test_a_single_unnamed_dataset_is_saved_as_train(tmp_path: Path, source: Path) -> None:
    _pipeline(tmp_path, source, split="test", fractions=None).preprocess_all()

    assert _saved(tmp_path / "out") == {"train": ["f", "g"]}


def test_fractions_split_a_single_dataset_into_their_names(tmp_path: Path, source: Path) -> None:
    _pipeline(
        tmp_path, source, split="train", fractions={"train": 0.75, "holdout": 0.25}
    ).preprocess_all()

    saved = _saved(tmp_path / "out")
    assert sorted(saved) == ["holdout", "train"]
    assert sorted(saved["train"] + saved["holdout"]) == SPLITS["train"]
    assert len(saved["holdout"]) == 1


def test_fractions_applied_to_several_input_splits_collide_before_anything_overlaps(
        tmp_path: Path, source: Path
) -> None:
    pipeline = _pipeline(tmp_path, source, split=None, fractions={"train": 1.0})

    with pytest.raises(ValueError, match="already wrote"):
        pipeline.preprocess_all()

    assert sorted(d.name for d in (tmp_path / "out").iterdir() if d.is_dir()) == ["train"]


def test_fractions_that_do_not_sum_to_one_are_rejected(tmp_path: Path, source: Path) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        _pipeline(tmp_path, source, split=None, fractions={"train": 0.5})


def test_the_output_records_the_rendered_config_and_cli(tmp_path: Path, source: Path) -> None:
    pipeline = _pipeline(tmp_path, source, split=None, fractions=None)
    pipeline.run()

    rendered = yaml.safe_load((tmp_path / "out" / RunRecord.CONFIG_FILE).read_text())
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())

    assert rendered["pipeline"]["rename_text"]["type"] == "ColumnRenamingPreprocessorConfig"
    assert provenance["config"] == rendered
    assert provenance["cli"] == (tmp_path / "out" / RunRecord.CLI_FILE).read_text().strip()
    assert PipelineDefinition.from_dict(rendered) == pipeline.state.definition
