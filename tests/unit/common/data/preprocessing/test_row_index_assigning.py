import pickle
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_from_disk

from larry.common.config.data.preprocessing.preprocessor_configs import RowIndexAssigningPreprocessorConfig
from larry.common.data.preprocessing.preprocessors import RowIndexAssigningPreprocessor
from tests.helpers.common import provenance


def _preprocessor(output_column: str) -> RowIndexAssigningPreprocessor:
    return RowIndexAssigningPreprocessor(provenance(), RowIndexAssigningPreprocessorConfig(output_column=output_column))


@pytest.fixture
def many_files_dataset(tmp_path: Path) -> Dataset:
    parts = []
    for i in range(40):
        Dataset.from_dict({"a": list(range(i * 500, (i + 1) * 500))}).save_to_disk(tmp_path / f"part{i}")
        parts.append(load_from_disk(tmp_path / f"part{i}"))
    return concatenate_datasets(parts)


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


def test_dataset_spread_over_many_files_stays_small_when_pickled_for_a_worker(many_files_dataset: Dataset) -> None:
    out = _preprocessor("row").preprocess_dataset(many_files_dataset)

    assert isinstance(out, Dataset)
    assert out["row"] == list(range(20_000))
    assert len(pickle.dumps(out)) < 20_000 * 8 + 50_000


def test_dataset_spread_over_many_files_is_rebuilt_quickly_by_a_worker_after_later_steps(
        many_files_dataset: Dataset
) -> None:
    out = _preprocessor("row").preprocess_dataset(many_files_dataset).rename_column("a", "b")
    worker_copy = pickle.loads(pickle.dumps(out.shard(8, 3, contiguous=True)))

    assert worker_copy["row"] == list(range(7_500, 10_000))
    assert worker_copy["b"] == list(range(7_500, 10_000))
    assert {f["filename"] for f in worker_copy.cache_files} <= {f["filename"] for f in many_files_dataset.cache_files}


def test_rows_follow_an_indices_mapping() -> None:
    dataset = Dataset.from_dict({"a": list(range(10))}).filter(lambda r: r["a"] % 3 == 0)

    out = _preprocessor("row").preprocess_dataset(dataset)

    assert out["a"] == [0, 3, 6, 9]
    assert out["row"] == [0, 1, 2, 3]


def test_row_column_fingerprint_is_stable_across_runs(tmp_path: Path) -> None:
    Dataset.from_dict({"a": list(range(100))}).save_to_disk(tmp_path / "source")

    first = _preprocessor("row").preprocess_dataset(load_from_disk(tmp_path / "source"))
    second = _preprocessor("row").preprocess_dataset(load_from_disk(tmp_path / "source"))

    assert isinstance(first, Dataset) and isinstance(second, Dataset)
    assert first._fingerprint == second._fingerprint
