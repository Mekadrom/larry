import torch

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.data.dataloading.voice_dataloading import RandomCropCollator
from larry.voice.model.vocoder import LarryVocoder

TINY = LarryVocoderConfig(d_model=32, d_inner=64, n_layers=2)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    items = [{"waveform": torch.randn(n) * 0.1, "source": 0} for n in (40_000, 9_000)]
    return RandomCropCollator(crop_samples=16_000, crops_per_clip=2, seed=0)(items)


def test_a_training_step_gives_a_finite_loss_and_gradients() -> None:
    torch.manual_seed(0)
    model = LarryVocoder(TINY)
    batch = _batch()
    mels, mel_lengths = model.frontend(batch["waveforms"], batch["waveform_lengths"])
    spec = model.codec.encode(batch["waveforms"])

    output = model(spec, mels, mel_lengths)
    loss = ConditionalFlowMatchingCriterion(CriterionConfig())(output)
    loss.backward()

    assert spec.shape == (3, TINY.stft_n_fft + 2, spec.shape[-1])
    assert spec.shape[-1] in (2 * mels.shape[-1] - 1, 2 * mels.shape[-1])
    assert output.logits.shape == output.targets.shape == spec.shape
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)


def test_padded_frames_do_not_contribute_to_the_loss() -> None:
    torch.manual_seed(0)
    model = LarryVocoder(TINY)
    batch = _batch()
    mels, mel_lengths = model.frontend(batch["waveforms"], batch["waveform_lengths"])
    output = model(model.codec.encode(batch["waveforms"]), mels, mel_lengths)
    criterion = ConditionalFlowMatchingCriterion(CriterionConfig())

    before = criterion.sample_losses(output, output.targets)
    output.targets[2, :, int(model.spec_lengths(mel_lengths, output.targets.shape[-1])[2]):] += 100.0

    assert torch.equal(criterion.sample_losses(output, output.targets), before)


def test_generation_returns_trimmed_waveforms() -> None:
    torch.manual_seed(0)
    model = LarryVocoder(TINY).eval()
    batch = _batch()
    mels, mel_lengths = model.frontend(batch["waveforms"], batch["waveform_lengths"])

    out = model.generate(mels, mel_lengths, steps=2, generator=torch.Generator().manual_seed(0))

    assert out.waveforms.shape[0] == 3
    assert out.waveform_lengths.tolist() == (mel_lengths * TINY.hop_length - TINY.stft_n_fft // 2).tolist()
    assert (out.waveform_lengths <= batch["waveform_lengths"]).all()
    assert torch.isfinite(out.waveforms).all()
