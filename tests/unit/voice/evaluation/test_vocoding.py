import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_model

from larry.common.model.pretrained import load_pretrained
from larry.voice.config.evaluation.evaluation_configs import PretrainedVocoderConfig
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.evaluation.vocoding import MelVocoder
from larry.voice.model.vocoder import LarryVocoder
from tests.unit.voice.model.test_speech_decoder import tiny_config

TINY = LarryVocoderConfig(d_model=32, d_inner=64, n_layers=2)


def _vocoder() -> LarryVocoder:
    torch.manual_seed(0)
    vocoder = LarryVocoder(TINY)
    for parameter in vocoder.parameters():
        parameter.data.normal_(0, 0.02)
    return vocoder


def _assert_same_weights(a: LarryVocoder, b: LarryVocoder) -> None:
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True))


def test_training_checkpoints_load_with_the_config_from_their_run_directory(tmp_path: Path) -> None:
    vocoder = _vocoder()
    (tmp_path / "config.json").write_text(json.dumps(vocoder.config.to_dict()))
    (tmp_path / "checkpoint-10").mkdir()
    save_model(vocoder, str(tmp_path / "checkpoint-10" / "model.safetensors"))

    loaded = load_pretrained(LarryVocoder, str(tmp_path / "checkpoint-10"))

    assert loaded.config == vocoder.config
    _assert_same_weights(loaded, vocoder)


def test_saved_pretrained_directories_still_load(tmp_path: Path) -> None:
    vocoder = _vocoder()
    vocoder.save_pretrained(tmp_path / "vocoder")

    _assert_same_weights(load_pretrained(LarryVocoder, str(tmp_path / "vocoder")), vocoder)


def test_a_configured_vocoder_is_frozen_and_renders_seeded_trimmed_clips(tmp_path: Path) -> None:
    _vocoder().save_pretrained(tmp_path / "vocoder")
    vocoder = MelVocoder.from_config(
        PretrainedVocoderConfig(model_path=str(tmp_path / "vocoder"), steps=2), torch.device("cpu")
    )
    waveforms = torch.randn(2, 8000) * 0.1
    lengths = torch.tensor([8000, 4800])
    mels, mel_lengths = vocoder.frontend(waveforms, lengths)

    clips = vocoder.clips(mels, mel_lengths)

    assert not any(p.requires_grad for p in vocoder.vocoder.parameters()) and not vocoder.vocoder.training
    assert [len(c) for c in clips] == (mel_lengths * TINY.hop_length - TINY.stft_n_fft // 2).tolist()
    assert all(torch.equal(a, b) for a, b in zip(clips, vocoder.clips(mels, mel_lengths), strict=True))
    assert vocoder.sample_rate == TINY.sample_rate
    assert vocoder.frame_seconds == TINY.hop_length / TINY.sample_rate


def test_decoders_must_share_the_vocoders_mels() -> None:
    vocoder = MelVocoder(_vocoder(), steps=2)

    vocoder.check_compatible(tiny_config())
    with pytest.raises(ValueError, match="n_mels"):
        vocoder.check_compatible(tiny_config(n_mels=100))
