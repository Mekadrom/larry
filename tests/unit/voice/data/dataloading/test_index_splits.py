from pathlib import Path

import datasets.config
import pytest

from larry.common.data.dataloading.indexed_dataset import index_split_files, load_index
from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from tests.helpers.common import write_parquet_splits

SPLITS = {
    "train.clean": ["c0", "c1"],
    "train.other": ["o0"],
    "dev.clean": ["d0"],
    "test.other": ["t0"],
}


@pytest.fixture
def index(tmp_path: Path) -> Path:
    return write_parquet_splits(tmp_path / "index", {name: {"id": ids} for name, ids in SPLITS.items()})


def test_split_names_are_matched_to_directories_exactly(index: Path) -> None:
    assert [Path(f).parent.name for f in index_split_files(str(index), "train.clean")] == ["train.clean"]


def test_plus_joins_splits_in_order(index: Path) -> None:
    files = index_split_files(str(index), "train.other+train.clean")

    assert [Path(f).parent.name for f in files] == ["train.other", "train.clean"]


def test_a_missing_split_is_named_in_the_error(index: Path) -> None:
    with pytest.raises(FileNotFoundError, match="'dev.other'"):
        index_split_files(str(index), "dev.clean+dev.other")


def test_load_index_reads_dotted_split_directories(
        tmp_path: Path, index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf")

    loaded = load_index(SpeechSourceConfig(name="hifi", index_path=str(index), split="train.clean+train.other"))

    assert loaded["id"] == ["c0", "c1", "o0"]
