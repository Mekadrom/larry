import csv
import os
from collections import defaultdict
from pathlib import Path

import duckdb
import pytest

from larry.common.config.training import build_trainer_config
from larry.common.utils import load_yaml_dict
from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.config.training.voice_trainer_configs import SpeechEncoderTrainerConfig
from larry.common.data.dataloading.indexed_dataset import index_split_files
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from tests.helpers.common import import_larry_submodules


@pytest.fixture(scope="session")
def trainer_config() -> SpeechEncoderTrainerConfig:
    config_file = os.environ.get("SPEAKER_MAPPING_CONFIG")
    if config_file is None:
        pytest.skip("SPEAKER_MAPPING_CONFIG not set")
    import_larry_submodules()
    definition = load_yaml_dict(Path(config_file).expanduser().resolve())
    config = build_trainer_config(definition.get("trainer", {}))
    assert isinstance(config, SpeechEncoderTrainerConfig)
    return config


@pytest.fixture(scope="session")
def sources(trainer_config: SpeechEncoderTrainerConfig) -> list[SpeechSourceConfig]:
    return trainer_config.mixture_config.train_sources + trainer_config.mixture_config.eval_sources


@pytest.fixture(scope="session")
def rows(trainer_config: SpeechEncoderTrainerConfig) -> list[tuple[str, int]]:
    path = Path(trainer_config.speaker_mapping_path).expanduser()
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["key", "id"]
        return [(row["key"], int(row["id"])) for row in reader]


@pytest.fixture(scope="session")
def expected(sources: list[SpeechSourceConfig]) -> dict[str, set[str]]:
    speakers: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        namespace = SpeakerMapping.namespace(source)
        if source.speaker_id_column is None:
            speakers[namespace].add(None)
            continue
        files = index_split_files(source.index_path, source.split)
        found = duckdb.execute(
            f'SELECT DISTINCT CAST("{source.speaker_id_column}" AS VARCHAR) FROM read_parquet(?) '
            f'WHERE "{source.speaker_id_column}" IS NOT NULL AND CAST("{source.speaker_id_column}" AS VARCHAR) <> \'\'',
            [files],
        ).fetchall()
        speakers[namespace].update(speaker for (speaker,) in found)
    return speakers


@pytest.fixture(scope="session")
def mapped(rows: list[tuple[str, int]]) -> dict[str, set[str]]:
    speakers: dict[str, set[str]] = defaultdict(set)
    for key, _ in rows:
        namespace, _, speaker = key.partition(":")
        speakers[namespace].add(speaker)
    return speakers


def test_ids_are_dense_and_keys_unique(rows: list[tuple[str, int]]) -> None:
    assert sorted(id_ for _, id_ in rows) == list(range(len(rows)))
    assert len({key for key, _ in rows}) == len(rows)


def test_keys_have_a_namespace_and_a_speaker(rows: list[tuple[str, int]]) -> None:
    malformed = [key for key, _ in rows if not all(key.partition(":")[0::2])]
    assert malformed == []


def test_namespaces_match_the_configured_sources(expected: dict[str, set[str]], mapped: dict[str, set[str]]) -> None:
    assert set(mapped) == set(expected)


def test_single_speaker_namespaces_have_one_id(expected: dict[str, set[str]], mapped: dict[str, set[str]]) -> None:
    for namespace, speakers in expected.items():
        if None in speakers:
            assert len(mapped[namespace]) == 1, namespace


def test_every_indexed_speaker_is_mapped_and_nothing_else(
        expected: dict[str, set[str]], mapped: dict[str, set[str]]
) -> None:
    for namespace, speakers in expected.items():
        if None in speakers:
            continue
        assert mapped[namespace] - speakers == set(), namespace
        assert speakers - mapped[namespace] == set(), namespace
