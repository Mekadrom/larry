from pathlib import Path

import torch

from larry.voice.model.speech_encoder import SpeechEncoderModel
from tests.helpers.voice import tiny_speech_encoder_config


def test_save_pretrained_round_trips_config_and_weights(tmp_path: Path) -> None:
    config = tiny_speech_encoder_config(40, mlp_dropout_p=0.0)
    model = SpeechEncoderModel(config)

    model.save_pretrained(tmp_path)
    loaded = SpeechEncoderModel.from_pretrained(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["README.md", "config.json", "model.safetensors"]
    assert loaded.config == config
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[k], restored[k]) for k in original)
