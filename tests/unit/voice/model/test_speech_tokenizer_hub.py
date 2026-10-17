from pathlib import Path

import torch

from larry.voice.config.model.voice_model_configs import SpeechEncoderModelConfig
from larry.voice.model.speech_tokenizer import SpeechEncoderModel


def test_save_pretrained_round_trips_config_and_weights(tmp_path: Path) -> None:
    config = SpeechEncoderModelConfig(
        d_model=32, n_heads=4, n_query_groups=4, rotary_embedding_dim=8,
        n_encoder1_layers=1, n_encoder2_layers=1, text_vocab_size=40, mlp_dropout_p=0.0,
    )
    model = SpeechEncoderModel(config)

    model.save_pretrained(tmp_path)
    loaded = SpeechEncoderModel.from_pretrained(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["README.md", "config.json", "model.safetensors"]
    assert loaded.config == config
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[k], restored[k]) for k in original)
