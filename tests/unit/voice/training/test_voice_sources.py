import io
import wave

import numpy as np
import pytest
import torch
from datasets import Audio, Dataset

from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.training.voice_trainers import SpeechDecoderTrainer, SpeechEncoderTrainer, VocoderTrainer, VoiceTrainer

SAMPLE_RATE = 16_000


def _wav(seconds: float = 0.25) -> bytes:
    samples = (np.sin(np.linspace(0, 200 * np.pi, int(SAMPLE_RATE * seconds))) * 12_000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _decoding_source() -> Dataset:
    source = Dataset.from_dict({"id": ["a"], "duration_s": [0.25], "audio": [{"bytes": _wav(), "path": None}]})
    return source.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


@pytest.mark.parametrize("trainer_type", [VocoderTrainer, SpeechEncoderTrainer, SpeechDecoderTrainer])
def test_every_voice_trainer_reads_audio_sources_as_raw_bytes(trainer_type: type[VoiceTrainer]) -> None:
    assert trainer_type.transform_as_source is VoiceTrainer.transform_as_source


def test_a_vocoder_source_yields_waveforms_instead_of_decoders() -> None:
    config = SpeechSourceConfig(name="clips", index_path="unused", source_path="unused", speaker_id_column=None)
    source = VoiceTrainer.transform_as_source(object.__new__(VocoderTrainer), config, _decoding_source())
    index = Dataset.from_dict({"row": [0], "id": ["a"], "duration_s": [0.25]})

    item = IndexedSpeechDataset([config], [index], [source], SAMPLE_RATE)[0]

    assert isinstance(source[0]["audio"]["bytes"], bytes)
    assert isinstance(item["waveform"], torch.Tensor) and item["waveform"].shape == (4000,)
    assert item["waveform"].abs().max() > 0.1
