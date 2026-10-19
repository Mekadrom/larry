import random
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from larry.voice.config.data.voice_data_configs import ASRSourceConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedASRDataset
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from larry.voice.training.voice_trainers import SpeechEncoderTrainer
from tests.helpers.audio import wav
from tests.helpers.common import uninitialized
from tests.helpers.voice import clip_dataset, toy_source


def test_source_ids_and_durations_cover_every_index_in_order_with_full_precision() -> None:
    a = Dataset.from_dict({"duration_s": [1.25, 2.5], "audio": [b"x", b"y"]})
    b = Dataset.from_dict({"duration_s": [0.1, 3.3, 7.123456789], "audio": [b"x", b"y", b"z"]})
    b = b.filter(lambda row: row["duration_s"] > 0.2)
    dataset = IndexedASRDataset(
        [ASRSourceConfig(name="a", index_path="a"), ASRSourceConfig(name="b", index_path="b")],
        [a, b],
        [None, None],
        sample_rate=16000,
    )

    source_ids, durations = dataset.source_ids_and_durations()

    assert source_ids.tolist() == [0, 0, 1, 1]
    assert durations.dtype == torch.float64
    assert durations.tolist() == [1.25, 2.5, 3.3, 7.123456789]


@pytest.mark.parametrize(("factor", "expected_samples"), [(1.0, 16000), (1.1, 14546), (0.9, 17778)])
def test_speed_perturbation_rescales_clip_length(factor: float, expected_samples: int) -> None:
    waveform = clip_dataset([("HI", 1.0)], speed_factors=(factor,))[0]["waveform"]

    assert abs(waveform.shape[-1] - expected_samples) <= 1


def test_speed_perturbation_picks_among_the_configured_factors() -> None:
    dataset = clip_dataset([("HI", 1.0)], speed_factors=(0.9, 1.1))

    lengths = {dataset[0]["waveform"].shape[-1] for _ in range(20)}

    assert lengths == {14546, 17778}


def _speed_lengths(dataset: IndexedASRDataset, draws: int = 12) -> list[int]:
    return [dataset[0]["waveform"].shape[-1] for _ in range(draws)]


def test_speed_perturbation_follows_the_seed_and_ignores_the_global_rng() -> None:
    random.seed(1)
    first_run = _speed_lengths(clip_dataset([("HI", 1.0)], speed_factors=(0.9, 1.1), seed=3))
    random.seed(2)
    second_run = _speed_lengths(clip_dataset([("HI", 1.0)], speed_factors=(0.9, 1.1), seed=3))

    assert first_run == second_run


@pytest.mark.parametrize("other", [{"seed": 4}, {"process_index": 1}])
def test_speed_perturbation_differs_across_seeds_and_processes(other: dict[str, int]) -> None:
    base = _speed_lengths(clip_dataset([("HI", 1.0)], speed_factors=(0.9, 1.1), seed=3), draws=24)
    changed = _speed_lengths(clip_dataset([("HI", 1.0)], speed_factors=(0.9, 1.1), **{"seed": 3, **other}), draws=24)

    assert base != changed


def test_speed_factors_reach_only_the_training_split() -> None:
    trainer = uninitialized(SpeechEncoderTrainer, _config=SimpleNamespace(
        speed_augment_factors=[0.9, 1.0, 1.1],
        seed=11,
        mixture_config=SimpleNamespace(sample_rate=16000),
    ))
    source = toy_source()
    index = Dataset.from_dict({"duration_s": [1.0]})

    train = trainer.load_indexed([source], [index], [None], is_eval=False)
    validation = trainer.load_indexed([source], [index], [None], is_eval=True)

    assert train.speed_factors == (0.9, 1.0, 1.1)
    assert validation.speed_factors == (1.0,)
    assert train.seed == 11


def test_speaker_keys_follow_row_order_and_mapping_keys() -> None:
    vctk = Dataset.from_dict({"speaker_id": ["p2", "p1"], "duration_s": [1.0, 1.0]})
    ls = Dataset.from_dict({"speaker_id": [0, 84], "duration_s": [1.0, 1.0]})
    lj = Dataset.from_dict({"duration_s": [1.0, 1.0, 1.0]})
    configs = [
        ASRSourceConfig(name="vctk", index_path="v"),
        ASRSourceConfig(name="heavy-test", index_path="h", speaker_namespace="heavy"),
        ASRSourceConfig(name="lj", index_path="l", speaker_id_column=None),
    ]
    dataset = IndexedASRDataset(configs, [vctk, ls, lj], [None, None, None], sample_rate=16000)

    keys = dataset.speaker_keys()

    assert keys[:4] == ["vctk:p2", "vctk:p1", "heavy:0", "heavy:84"]
    assert keys[4:] == [SpeakerMapping.key(configs[2], None)] * 3
    assert len(keys) == len(dataset)


def test_speaker_keys_respect_an_index_filter() -> None:
    index = Dataset.from_dict({"speaker_id": ["a", "b", "c"], "duration_s": [1.0, 0.1, 1.0]})
    index = index.filter(lambda row: row["duration_s"] > 0.5)
    dataset = IndexedASRDataset([ASRSourceConfig(name="s", index_path="s")], [index], [None], sample_rate=16000)

    assert dataset.speaker_keys() == ["s:a", "s:c"]


def _minified(rows: list[int], ids: list[str]) -> Dataset:
    return Dataset.from_dict({"row": rows, "id": ids})


def _source(clips: list[tuple[str, str, float, str]]) -> Dataset:
    return Dataset.from_dict({
        "id": [c[0] for c in clips],
        "audio": [{"bytes": wav(c[2], 16000), "path": f"{c[0]}.wav"} for c in clips],
        "text_normalized": [c[1] for c in clips],
        "duration_s": [c[2] for c in clips],
        "speaker_id": [c[3] for c in clips],
    })


SOURCE_CLIPS = [("a", "ONE", 0.5, "s1"), ("b", "TWO", 1.0, "s2"), ("c", "THREE", 1.5, "s1")]


def test_a_minified_index_reads_every_column_from_its_source_row() -> None:
    dataset = IndexedASRDataset(
        [toy_source()], [_minified([2, 0], ["c", "a"])], [_source(SOURCE_CLIPS)], sample_rate=16000
    )

    assert [dataset[i]["text"] for i in range(2)] == ["THREE", "ONE"]
    assert dataset[0]["waveform"].shape[-1] == 24000
    assert dataset.source_ids_and_durations()[1].tolist() == [1.5, 0.5]
    assert dataset.speaker_keys() == ["toy:s1", "toy:s1"]


def test_index_columns_take_precedence_over_the_source() -> None:
    index = Dataset.from_dict({"row": [1], "id": ["b"], "text_normalized": ["DEUX"], "duration_s": [9.0]})
    dataset = IndexedASRDataset([toy_source()], [index], [_source(SOURCE_CLIPS)], sample_rate=16000)

    assert dataset[0]["text"] == "DEUX"
    assert dataset.source_ids_and_durations()[1].tolist() == [9.0]
    assert dataset.speaker_keys() == ["toy:s2"]


def test_source_rows_follow_a_filtered_source() -> None:
    source = _source([("x", "SKIP", 0.1, "s0"), *SOURCE_CLIPS]).filter(lambda r: r["id"] != "x")
    dataset = IndexedASRDataset([toy_source()], [_minified([1], ["b"])], [source], sample_rate=16000)

    assert dataset[0]["text"] == "TWO"
    assert dataset.source_ids_and_durations()[1].tolist() == [1.0]


def test_a_row_pointing_at_another_id_is_rejected() -> None:
    dataset = IndexedASRDataset([toy_source()], [_minified([1], ["a"])], [_source(SOURCE_CLIPS)], sample_rate=16000)

    with pytest.raises(KeyError, match="row 1 is b"):
        dataset[0]


def test_a_minified_index_without_a_source_names_the_missing_column() -> None:
    dataset = IndexedASRDataset([toy_source()], [_minified([0], ["a"])], [None], sample_rate=16000)

    with pytest.raises(KeyError, match="duration_s"):
        dataset.source_ids_and_durations()
