import math

import pytest
import torch
from torch import nn

from larry.common.config.config import ConfigValues
from larry.common.model.init import WeightInitializer
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig, SpeechEncoderModelConfig
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.vocoder import LarryVocoder
from tests.helpers.config_recipes import CONFIG_ROOT
from tests.helpers.voice import tiny_speech_encoder
from tests.unit.voice.model.test_speech_decoder import tiny_config


def _wide(**overrides: object) -> dict[str, object]:
    return {"d_model": 256, "n_heads": 4, "n_query_groups": 4, "rotary_embedding_dim": 16, **overrides}


def test_vocoder_starts_as_the_identity_with_small_truncated_weights() -> None:
    torch.manual_seed(0)
    vocoder = LarryVocoder(LarryVocoderConfig(d_model=256, d_inner=512, n_layers=2))

    block = vocoder.backbone.blocks[0]
    assert not vocoder.backbone.out.weight.any() and not vocoder.backbone.out.bias.any()
    assert not block.norm.modulation[1].weight.any()
    assert block.pointwise1.weight.abs().max() <= 0.04 + 1e-6
    assert block.pointwise1.weight.std().item() == pytest.approx(0.02 * 0.88, rel=0.1)
    assert not block.pointwise1.bias.any()


def test_speech_decoder_uses_dit_style_init() -> None:
    torch.manual_seed(0)
    decoder = SpeechDecoderModel(tiny_config(**_wide()))

    embed = decoder.backbone.embed
    fan_in, fan_out = embed.weight.shape[1] * embed.weight.shape[2], embed.weight.shape[0] * embed.weight.shape[2]
    assert embed.weight.abs().max() <= math.sqrt(6 / (fan_in + fan_out)) + 1e-6
    assert not decoder.backbone.out.weight.any()
    assert not decoder.backbone.blocks[0].attn_norm.modulation[1].weight.any()
    assert not decoder.speaker_embedding.logvar_embedding.weight.any()
    assert decoder.speaker_embedding.mean_embedding.weight.std().item() == pytest.approx(0.02, rel=0.3)
    assert decoder.backbone.time.mlp[0].weight.std().item() == pytest.approx(0.02, rel=0.1)


def test_speech_encoder_keeps_pytorch_init_by_default() -> None:
    encoder = tiny_speech_encoder(16)

    assert encoder.config.weight_init == "pytorch"
    assert encoder.ctc_head.weight.any()
    bound = 1 / math.sqrt(encoder.config.d_model)
    assert encoder.ctc_head.weight.abs().max() <= bound + 1e-6


def test_speech_encoder_scaled_normal_init() -> None:
    torch.manual_seed(0)
    encoder = tiny_speech_encoder(16, weight_init="scaled_normal", enable_fsq_ctc_head=True, **_wide())

    d_model = encoder.config.d_model
    assert not encoder.ctc_head.weight.any() and not encoder.fsq_ctc_head.weight.any()
    assert encoder.fsq.down.weight.std().item() == pytest.approx(1 / math.sqrt(d_model), rel=0.1)
    conv1 = encoder.encoder1.conv1
    fan_in = conv1.weight.shape[1] * conv1.weight.shape[2]
    assert conv1.weight.std().item() == pytest.approx(math.sqrt(2 / fan_in), rel=0.1)
    block = encoder.encoder2[0]
    assert block.self_attn.q_proj.weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert block.self_attn.o_proj.weight.std().item() == pytest.approx(0.02 / math.sqrt(2), rel=0.1)


def test_the_scaled_init_overlay_turns_it_on() -> None:
    values = ConfigValues([
        CONFIG_ROOT / "training/voice/ctc/larry-speech-encoder.yaml",
        CONFIG_ROOT / "training/voice/ctc/overlays/scaled_init.yaml",
    ]).merged()

    config = SpeechEncoderModelConfig.from_dict({k: v for k, v in values["model"].items() if k != "type"})

    assert (config.weight_init, config.init_std) == ("scaled_normal", 0.02)


def test_residual_outputs_cover_attention_fsmn_memory_and_mlp() -> None:
    encoder = tiny_speech_encoder(16)

    fsmn_block = encoder.encoder1.blocks[0]
    outputs = WeightInitializer.residual_outputs(fsmn_block)

    assert outputs == [fsmn_block.attn.o_proj, fsmn_block.attn.fsmn, fsmn_block.mlp[-1]]
    assert WeightInitializer.residual_outputs(nn.Linear(2, 2)) == []
