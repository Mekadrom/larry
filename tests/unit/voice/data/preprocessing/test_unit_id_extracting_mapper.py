from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Audio, Dataset, List, Value
from datasets.fingerprint import Hasher

from larry.voice.config.preprocessing.voice_mapper_configs import LarrySpeechEncoderUnitIdExtractingMapperConfig
from larry.voice.data.preprocessing import voice_mappers
from larry.voice.data.preprocessing.voice_mappers import LarrySpeechEncoderUnitIdExtractingMapper
from larry.voice.model.speech_encoder import SpeechEncoderModel
from larry.voice.utils import bytes_to_waveforms
from tests.helpers.audio import wav
from tests.helpers.common import provenance
from tests.helpers.voice import tiny_speech_encoder_config

TINY_ENCODER = tiny_speech_encoder_config(8, fsq_dims=4, fsq_levels=3)


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    torch.manual_seed(0)
    path = tmp_path / "encoder"
    SpeechEncoderModel(TINY_ENCODER).save_pretrained(path)
    monkeypatch.setitem(voice_mappers._ENCODERS, str(path), SpeechEncoderModel.from_pretrained(path))
    return str(path)


def _mapper(checkpoint: str, **overrides: object) -> LarrySpeechEncoderUnitIdExtractingMapper:
    config = LarrySpeechEncoderUnitIdExtractingMapperConfig(
        **{"speech_encoder_checkpoint_path": checkpoint, "num_proc": 1, "cache_results": False, **overrides}
    )
    return LarrySpeechEncoderUnitIdExtractingMapper(provenance(), config)


def _audio(seconds: float, frequency: float = 300) -> dict[str, object]:
    return {"bytes": wav(seconds, 16000, frequency=frequency), "path": None}


def _reference(checkpoint: str, audio: dict[str, object]) -> np.ndarray:
    model = voice_mappers._ENCODERS[checkpoint]
    waveform = bytes_to_waveforms(audio["bytes"], 16000)[0]
    with torch.inference_mode():
        unit_ids, frame_lengths = model.tokenize(waveform[None], torch.tensor([len(waveform)]))
    return unit_ids[0, :int(frame_lengths[0])].numpy()


def test_each_example_gets_a_1d_int16_array_cropped_to_its_own_frame_length(checkpoint: str) -> None:
    out = _mapper(checkpoint).preprocess_batch({"audio": [_audio(2.0), _audio(0.5), _audio(1.2)]})

    unit_ids, frame_lengths = out["unit_ids"], out["frame_lengths"]
    assert [u.ndim for u in unit_ids] == [1, 1, 1]
    assert {u.dtype for u in unit_ids} == {np.dtype(np.int16)}
    assert [len(u) for u in unit_ids] == list(frame_lengths)
    assert frame_lengths[0] > frame_lengths[2] > frame_lengths[1]


def test_unit_ids_match_tokenizing_each_clip_alone(checkpoint: str) -> None:
    clips = [_audio(2.0, 220), _audio(0.5, 440), _audio(1.2, 330)]

    out = _mapper(checkpoint).preprocess_batch({"audio": clips})

    for clip, unit_ids in zip(clips, out["unit_ids"], strict=True):
        np.testing.assert_array_equal(unit_ids, _reference(checkpoint, clip))


def test_unit_ids_lie_in_the_fsq_codebook(checkpoint: str) -> None:
    out = _mapper(checkpoint).preprocess_batch({"audio": [_audio(1.0, 200), _audio(0.7, 900)]})

    ids = np.concatenate(out["unit_ids"])
    assert ids.min() >= 0
    assert ids.max() < TINY_ENCODER.n_units


def test_columns_follow_the_config() -> None:
    config = LarrySpeechEncoderUnitIdExtractingMapperConfig(
        speech_encoder_checkpoint_path="x",
        audio_column="clip",
        unit_ids_output_column="units",
        frame_lengths_output_column="n_units",
    )

    assert config.input_columns == ["clip"]
    assert config.output_columns == ["units", "n_units"]


def test_renamed_columns_are_read_and_written(checkpoint: str) -> None:
    mapper = _mapper(checkpoint, audio_column="clip", unit_ids_output_column="units", frame_lengths_output_column="n")

    out = mapper.preprocess_batch({"clip": [_audio(1.0)]})

    assert set(out) == {"units", "n"}
    assert len(out["units"][0]) == out["n"][0]


def test_mapping_a_dataset_keeps_other_columns_and_stores_typed_features(checkpoint: str) -> None:
    clips = [_audio(1.5), _audio(0.4), _audio(0.9)]
    dataset = Dataset.from_dict({"id": ["a", "b", "c"], "audio": clips}).cast_column("audio", Audio(decode=False))

    out = _mapper(checkpoint, num_proc=None).preprocess_dataset(dataset)

    assert isinstance(out, Dataset)
    assert out["id"] == ["a", "b", "c"]
    assert out.features["unit_ids"] == List(Value("int16"))
    assert out.features["frame_lengths"] == Value("int32")
    expected = _mapper(checkpoint).preprocess_batch({"audio": clips})
    assert out["unit_ids"] == [u.tolist() for u in expected["unit_ids"]]
    assert out["frame_lengths"] == [len(u) for u in out["unit_ids"]]


def test_mapper_fingerprint_does_not_change_once_the_encoder_is_loaded(checkpoint: str) -> None:
    mapper = _mapper(checkpoint)
    before = Hasher.hash(mapper._preprocess_columns)

    mapper.preprocess_batch({"audio": [_audio(0.5)]})

    assert Hasher.hash(mapper._preprocess_columns) == before


class _FakeEncoder:
    def __init__(self) -> None:
        self.devices: list[object] = []

    def to(self, device: object) -> _FakeEncoder:
        self.devices.append(device)
        return self


def test_each_checkpoint_is_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []

    def from_pretrained(path: str) -> _FakeEncoder:
        loaded.append(path)
        return _FakeEncoder()

    monkeypatch.setattr(voice_mappers, "_ENCODERS", {})
    monkeypatch.setattr(SpeechEncoderModel, "from_pretrained", from_pretrained)

    cpu = torch.device("cpu")
    first = voice_mappers._lazy_get_encoder("a", cpu)
    again = voice_mappers._lazy_get_encoder("a", cpu)
    other = voice_mappers._lazy_get_encoder("b", cpu)

    assert first is again
    assert other is not first
    assert loaded == ["a", "b"]


def test_the_encoder_is_placed_on_the_cpu_when_no_gpu_is_available(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "encoder"
    SpeechEncoderModel(TINY_ENCODER).save_pretrained(path)
    monkeypatch.setattr(voice_mappers, "_ENCODERS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    mapper = _mapper(str(path))

    out = mapper.preprocess_batch({"audio": [_audio(0.5)]})

    assert mapper.device == torch.device("cpu")
    assert next(voice_mappers._ENCODERS[str(path)].parameters()).device == torch.device("cpu")
    assert len(out["unit_ids"][0]) == out["frame_lengths"][0]


@pytest.mark.parametrize(("rank", "devices", "expected"), [(None, 4, "cuda:0"), (5, 4, "cuda:1"), (2, 1, "cuda:0")])
def test_mapper_device_follows_the_worker_rank(
        monkeypatch: pytest.MonkeyPatch, rank: int | None, devices: int, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    mapper = _mapper("x")
    monkeypatch.setattr(mapper, "preprocess_batch", lambda batch: batch)

    mapper._preprocess_batch({}, rank)

    assert mapper.device == torch.device(expected)
