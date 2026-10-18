import io
import math
import random
import struct
import wave
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from larry.voice.config.data.voice_data_configs import ASRSourceConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedASRDataset
from larry.voice.data.preprocessing.global_speaker_map import SpeakerMapping
from larry.voice.training.voice_trainers import SpeechEncoderTrainer


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


def _wav(seconds: float, sample_rate: int) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 300 * i / sample_rate)))
        for i in range(int(seconds * sample_rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return buffer.getvalue()


def _clip_dataset(speed_factors: tuple[float, ...]) -> IndexedASRDataset:
    index = Dataset.from_dict({
        "audio": [{"bytes": _wav(1.0, 16000), "path": "a.wav"}],
        "text_normalized": ["HI"],
        "duration_s": [1.0],
    })
    source = ASRSourceConfig(name="toy", index_path="toy")
    return IndexedASRDataset([source], [index], [None], sample_rate=16000, speed_factors=speed_factors)


@pytest.mark.parametrize(("factor", "expected_samples"), [(1.0, 16000), (1.1, 14546), (0.9, 17778)])
def test_speed_perturbation_rescales_clip_length(factor: float, expected_samples: int) -> None:
    waveform = _clip_dataset((factor,))[0]["waveform"]

    assert abs(waveform.shape[-1] - expected_samples) <= 1


def test_speed_perturbation_picks_among_the_configured_factors() -> None:
    dataset = _clip_dataset((0.9, 1.1))
    random.seed(0)

    lengths = {dataset[0]["waveform"].shape[-1] for _ in range(20)}

    assert lengths == {14546, 17778}


def test_speed_factors_reach_only_the_training_split() -> None:
    trainer = SpeechEncoderTrainer.__new__(SpeechEncoderTrainer)
    vars(trainer).update(
        _config=SimpleNamespace(
            speed_augment_factors=[0.9, 1.0, 1.1],
            mixture_config=SimpleNamespace(sample_rate=16000),
        ),
    )
    source = ASRSourceConfig(name="toy", index_path="toy")
    index = Dataset.from_dict({"duration_s": [1.0]})

    train = trainer.load_indexed([source], [index], [None], is_eval=False)
    validation = trainer.load_indexed([source], [index], [None], is_eval=True)

    assert train.speed_factors == (0.9, 1.0, 1.1)
    assert validation.speed_factors == (1.0,)


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
