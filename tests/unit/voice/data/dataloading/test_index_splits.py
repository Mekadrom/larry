from pathlib import Path

import datasets.config
import pytest
from datasets import Dataset

from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.common.data.dataloading.indexed_dataset import index_split_files
from larry.common.data.dataloading.indexed_dataset import load_index


def _write_index(root: Path, splits: dict[str, list[str]]) -> Path:
    for name, ids in splits.items():
        (root / name).mkdir(parents=True)
        Dataset.from_dict({"id": ids}).to_parquet(root / name / f"{name}-00000-of-00001.parquet")
    return root


SPLITS = {
    "train.clean": ["c0", "c1"],
    "train.other": ["o0"],
    "dev.clean": ["d0"],
    "test.other": ["t0"],
}


def test_split_names_are_matched_to_directories_exactly(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "index", SPLITS)

    assert [Path(f).parent.name for f in index_split_files(str(index), "train.clean")] == ["train.clean"]


def test_plus_joins_splits_in_order(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "index", SPLITS)

    files = index_split_files(str(index), "train.other+train.clean")

    assert [Path(f).parent.name for f in files] == ["train.other", "train.clean"]


def test_a_missing_split_is_named_in_the_error(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "index", SPLITS)

    with pytest.raises(FileNotFoundError, match="'dev.other'"):
        index_split_files(str(index), "dev.clean+dev.other")


def test_load_index_reads_dotted_split_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf")
    index = _write_index(tmp_path / "index", SPLITS)

    loaded = load_index(SpeechSourceConfig(name="hifi", index_path=str(index), split="train.clean+train.other"))

    assert loaded["id"] == ["c0", "c1", "o0"]
