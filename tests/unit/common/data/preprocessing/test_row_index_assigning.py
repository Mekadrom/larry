import pickle
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_from_disk

from larry.common.config.data.preprocessing.pipeline_config import Provenance, ProvenanceAncestry
from larry.common.config.data.preprocessing.preprocessor_configs import RowIndexAssigningPreprocessorConfig
from larry.common.data.preprocessing.preprocessors import RowIndexAssigningPreprocessor


def _preprocessor(output_column: str) -> RowIndexAssigningPreprocessor:
    provenance = Provenance(id_columns=[], ancestry=ProvenanceAncestry(datasets=set(), models=set()))
    return RowIndexAssigningPreprocessor(provenance, RowIndexAssigningPreprocessorConfig(output_column=output_column))


@pytest.mark.parametrize("output_column", ["row", "idx"])
def test_each_split_is_numbered_from_zero_under_the_configured_column(output_column: str) -> None:
    dataset = DatasetDict({
        "train": Dataset.from_dict({"a": [1, 2, 3]}),
        "test": Dataset.from_dict({"a": [4, 5]}),
    })

    out = _preprocessor(output_column).preprocess_dataset(dataset)

    assert isinstance(out, DatasetDict)
    assert {name: split[output_column] for name, split in out.items()} == {"train": [0, 1, 2], "test": [0, 1]}


@pytest.mark.parametrize("output_column", ["row", "idx"])
def test_single_dataset_is_numbered_in_its_current_order(output_column: str) -> None:
    dataset = Dataset.from_dict({"a": list(range(6))}).filter(lambda r: r["a"] % 2 == 0)

    out = _preprocessor(output_column).preprocess_dataset(dataset)

    assert isinstance(out, Dataset)
    assert out["a"] == [0, 2, 4]
    assert out[output_column] == [0, 1, 2]
    assert out.features[output_column] == Value("int64")


def test_dataset_spread_over_many_files_stays_small_when_pickled_for_a_worker(tmp_path: Path) -> None:
    parts = []
    for i in range(40):
        Dataset.from_dict({"a": list(range(i * 500, (i + 1) * 500))}).save_to_disk(tmp_path / f"part{i}")
        parts.append(load_from_disk(tmp_path / f"part{i}"))
    dataset = concatenate_datasets(parts)

    out = _preprocessor("row").preprocess_dataset(dataset)

    assert isinstance(out, Dataset)
    assert out["row"] == list(range(20_000))
    assert len(pickle.dumps(out)) < 200_000
