import dataclasses
from types import SimpleNamespace

import torch

from larry.voice.model.vocoder import LarryVocoder
from larry.voice.training.voice_trainers import VocoderTrainer
from tests.helpers.config_recipes import load_training
from tests.unit.voice.model.test_vocoder import TINY


def _vocoder(xt_shortcut: bool) -> LarryVocoder:
    torch.manual_seed(0)
    return LarryVocoder(dataclasses.replace(TINY, xt_shortcut=xt_shortcut))


def _inputs(model: LarryVocoder) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1)
    frames = 12
    xt, mels = torch.randn(2, 2 * model.n_bins, frames), torch.randn(2, TINY.n_mels, frames)
    return xt, torch.rand(2), mels, torch.ones(2, frames)


def test_the_shortcut_is_off_by_default_and_on_in_its_overlay() -> None:
    assert _vocoder(False).backbone.shortcut is None
    assert not load_training("larry-vocoder").model.xt_shortcut
    assert load_training("larry-vocoder+xt_shortcut").model.xt_shortcut


def test_a_new_shortcut_starts_as_the_baseline() -> None:
    plain, gated = _vocoder(False), _vocoder(True)
    gated.load_state_dict(plain.state_dict(), strict=False)

    with torch.no_grad():
        assert torch.equal(plain.backbone(*_inputs(plain)), gated.backbone(*_inputs(gated)))


def test_the_shortcut_scales_each_channel_of_xt_by_a_time_dependent_weight() -> None:
    model = _vocoder(True)
    shortcut = model.backbone.shortcut
    assert shortcut is not None and shortcut.weight.shape == (2 * model.n_bins, TINY.d_model)
    with torch.no_grad():
        shortcut.bias.fill_(-1.0)
    xt, t, mels, mask = _inputs(model)

    with torch.no_grad():
        velocity = model.backbone(xt, t, mels, mask)

    assert torch.allclose(velocity, -xt, atol=1e-6)


def test_the_shortcut_trains_and_stays_off_muon() -> None:
    model = _vocoder(True).train()
    xt, t, mels, mask = _inputs(model)
    with torch.no_grad():
        model.backbone.out.weight.normal_(std=0.02)

    model.backbone(xt, t, mels, mask).square().mean().backward()

    assert model.backbone.shortcut.bias.grad is not None and model.backbone.shortcut.bias.grad.abs().sum() > 0
    trainer = object.__new__(VocoderTrainer)
    trainer._config = SimpleNamespace(muon_first_layer_names=[], muon_last_layer_names=["backbone.out.weight"])
    excluded = trainer.muon_excluded_names(model)
    assert "backbone.shortcut.weight" in excluded and "backbone.out.weight" in excluded
    assert "backbone.shortcut.weight" not in trainer.muon_excluded_names(_vocoder(False))
