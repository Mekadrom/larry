import csv
import os
from collections import defaultdict
from pathlib import Path

import pytest

from larry.common.config.training.definition import TrainingDefinition
from larry.common.config.values import ConfigValues
from larry.common.data.dataloading.indexed_dataset import indexed_column, load_index, load_source_for_columns
from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.config.training.voice_trainer_configs import SpeechEncoderTrainerConfig
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from tests.helpers.common import import_larry_submodules


@pytest.fixture(scope="session")
def trainer_config() -> SpeechEncoderTrainerConfig:
    config_file = os.environ.get("SPEAKER_MAPPING_CONFIG")
    if config_file is None:
        pytest.skip("SPEAKER_MAPPING_CONFIG not set")
    import_larry_submodules()
    config = ConfigValues(config_file.split(os.pathsep)).load(TrainingDefinition).trainer
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
        index = load_index(source)
        column = indexed_column(index, load_source_for_columns(source, index, [source.speaker_id_column]),
                                source.speaker_id_column)
        speakers[namespace].update(str(s) for s in column.unique().to_pylist() if s is not None and str(s) != "")
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
