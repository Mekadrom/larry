from types import SimpleNamespace

import pytest
import torch

from larry.voice.config.criterion.voice_criterion_configs import VocoderCriterionConfig
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.config.training.voice_trainer_configs import MelAugmentConfig
from larry.voice.criteria.flow_matching import VocoderCriterion
from larry.voice.criteria.spectral import MultiResolutionSTFTLoss
from larry.voice.model.vocoder import LarryVocoder
from larry.voice.model.voice_models import LarryVocoderOutput
from larry.voice.model.voice_modules import MelAugmenter
from larry.voice.training.voice_trainers import VocoderTrainer
from tests.helpers.common import uninitialized

TINY = LarryVocoderConfig(d_model=32, d_inner=64, n_layers=2)


def _waveforms() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    t = torch.arange(16_000) / 16_000
    waveforms = torch.stack([0.3 * torch.sin(2 * torch.pi * 180 * t), 0.2 * torch.sin(2 * torch.pi * 260 * t)])
    waveforms[1, 12_000:] = 0
    return waveforms, torch.tensor([16_000, 12_000])


def _output(model: LarryVocoder, waveforms: torch.Tensor, lengths: torch.Tensor) -> LarryVocoderOutput:
    mels, mel_lengths = model.frontend(waveforms, lengths)
    return model(model.codec.encode(waveforms), mels, mel_lengths)


def _terms(config: VocoderCriterionConfig, model: LarryVocoder) -> tuple[dict, LarryVocoderOutput]:
    waveforms, lengths = _waveforms()
    mels, mel_lengths = model.frontend(waveforms, lengths)
    output = _output(model, waveforms, lengths)
    criterion = VocoderCriterion(config, model.config)
    terms = criterion.loss_terms(
        output, waveforms=waveforms, waveform_lengths=lengths, mels=mels, mel_lengths=mel_lengths
    )
    return terms, output


def test_one_step_estimate_with_the_true_velocity_recovers_the_waveform() -> None:
    model = LarryVocoder(TINY)
    waveforms, _ = _waveforms()
    spec = model.codec.encode(waveforms)
    xt, t, target = model.path(spec)

    estimates = model.one_step_waveforms(xt, t, target)

    n = estimates.shape[-1]
    assert n == (spec.shape[-1] - 1) * TINY.hop_length
    assert torch.allclose(estimates, waveforms[:, :n], atol=2e-3)


def test_spectral_terms_are_logged_without_gradients_at_zero_weight() -> None:
    torch.manual_seed(0)
    terms, _ = _terms(VocoderCriterionConfig(), LarryVocoder(TINY))

    assert list(terms) == ["flow", "stft", "mel"]
    assert terms["stft"].weight == terms["mel"].weight == 0.0
    assert not terms["stft"].sums.requires_grad and not terms["mel"].sums.requires_grad
    assert all(torch.isfinite(term.value) and term.value > 0 for term in terms.values())


def test_weighted_spectral_terms_reach_the_velocity_network() -> None:
    torch.manual_seed(0)
    model = LarryVocoder(TINY)
    torch.nn.init.normal_(model.backbone.out.weight, std=0.02)
    terms, _ = _terms(VocoderCriterionConfig(flow_weight=0.0, stft_weight=1.0, mel_weight=1.0), model)

    (terms["stft"].weight * terms["stft"].value + terms["mel"].weight * terms["mel"].value).backward()

    assert model.backbone.out.weight.grad is not None and model.backbone.out.weight.grad.abs().sum() > 0


def test_spectral_terms_are_weighted_by_t_to_the_configured_power() -> None:
    model = LarryVocoder(TINY)
    torch.manual_seed(0)
    flat, output = _terms(VocoderCriterionConfig(estimate_t_power=0.0), model)
    torch.manual_seed(0)
    squared, _ = _terms(VocoderCriterionConfig(estimate_t_power=2.0), model)

    t = output.t
    assert torch.allclose(squared["stft"].sums, flat["stft"].sums * t.square(), rtol=1e-4)
    assert torch.allclose(squared["mel"].sums, flat["mel"].sums * t.square(), rtol=1e-4)


def test_multi_resolution_stft_loss_ignores_padding() -> None:
    loss = MultiResolutionSTFTLoss([256, 512])
    waveforms, lengths = _waveforms()

    assert torch.allclose(loss(waveforms, waveforms, lengths), torch.zeros(2), atol=1e-6)

    noisy = waveforms + 0.05 * torch.randn_like(waveforms)
    padded = waveforms.clone()
    padded[1, 14_000:] = torch.randn(2_000)
    assert (loss(noisy, waveforms, lengths) > 0).all()
    assert loss(padded, waveforms, lengths)[0] == 0


def test_mel_augmenter_is_the_identity_when_disabled() -> None:
    mels = torch.randn(3, 8, 20)

    assert torch.equal(MelAugmenter(MelAugmentConfig())(mels), mels)


def test_mel_blurs_act_along_their_own_axis() -> None:
    time_constant = torch.randn(2, 8, 1).expand(2, 8, 20).contiguous()
    freq_constant = torch.randn(2, 1, 20).expand(2, 8, 20).contiguous()

    time_blur = MelAugmenter(MelAugmentConfig(time_blur_p=1.0, time_blur_sigma=2.0))
    freq_blur = MelAugmenter(MelAugmentConfig(freq_blur_p=1.0, freq_blur_sigma=2.0))

    assert torch.allclose(time_blur(time_constant), time_constant, atol=1e-6)
    assert not torch.allclose(time_blur(freq_constant), freq_constant)
    assert torch.allclose(freq_blur(freq_constant), freq_constant, atol=1e-6)
    assert not torch.allclose(freq_blur(time_constant), time_constant)


def test_mel_noise_is_drawn_per_row_up_to_the_configured_std() -> None:
    torch.manual_seed(0)
    mels = torch.zeros(64, 8, 200)

    noised = MelAugmenter(MelAugmentConfig(noise_p=0.5, noise_std=0.2))(mels)

    stds = noised.std(dim=(1, 2))
    assert 10 < int((stds == 0).sum()) < 54
    assert float(stds.max()) <= 0.2 * 1.1


def _trainer(model: LarryVocoder, augment: MelAugmentConfig) -> VocoderTrainer:
    return uninitialized(
        VocoderTrainer,
        _model=model,
        unwrapped_model=model,
        _config=SimpleNamespace(mel_augment=augment),
        criterion=VocoderCriterion(VocoderCriterionConfig(), model.config),
        accelerator=SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda m: m),
    )


@pytest.mark.parametrize("training", [True, False])
def test_trainer_augments_conditioning_only_in_training(training: bool) -> None:
    torch.manual_seed(0)
    model = LarryVocoder(TINY)
    model.train(training)
    trainer = _trainer(model, MelAugmentConfig(noise_p=1.0, noise_std=1.0))
    waveforms, lengths = _waveforms()
    seen: list[torch.Tensor] = []
    backbone = model.backbone.forward
    model.backbone.forward = lambda xt, t, mels, mask: seen.append(mels) or backbone(xt, t, mels, mask)

    output = trainer._forward_pass({"waveforms": waveforms, "waveform_lengths": lengths})

    clean, _ = model.frontend(waveforms, lengths)
    assert set(output.terms or {}) == {"flow", "stft", "mel"}
    assert torch.equal(seen[0], clean) != training
