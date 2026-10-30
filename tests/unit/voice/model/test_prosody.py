import math

import pytest
import torch

from larry.voice.model.prosody import F0RangePredictor, F0Statistics, HarmonicConditioning
from tests.unit.voice.model.test_speech_decoder import _batch
from tests.unit.voice.model.test_speech_decoder import _model as tiny_decoder


def _f0(frames: int = 40) -> torch.Tensor:
    f0 = 150.0 * torch.exp(0.1 * torch.sin(torch.arange(frames) / 3.0))
    f0[::7] = 0.0
    return f0[None]


def test_the_contour_is_speaker_invariant_and_zero_where_unvoiced() -> None:
    f0 = _f0()
    mask = torch.ones_like(f0, dtype=torch.bool)

    low, high = F0Statistics.from_f0(f0, mask), F0Statistics.from_f0(2.0 * f0, mask)

    voiced = f0 > 0
    assert torch.allclose(low.contour, high.contour, atol=1e-5)
    assert torch.equal(low.contour[~voiced], torch.zeros(int((~voiced).sum())))
    assert low.contour[voiced].mean().abs() < 1e-5 and (low.contour[voiced].std(unbiased=False) - 1).abs() < 1e-4
    assert torch.allclose(high.log_f0_mean - low.log_f0_mean, torch.tensor([math.log(2.0)]), atol=1e-5)


def test_padding_and_too_few_voiced_frames_give_no_range() -> None:
    f0 = torch.tensor([[200.0, 0.0, 0.0, 0.0], [200.0, 210.0, 220.0, 0.0]])
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])

    stats = F0Statistics.from_f0(f0, mask)

    assert stats.has_range.tolist() == [False, True]
    assert stats.voiced[1].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert torch.equal(stats.contour[0], torch.zeros(4))


def test_the_range_predictor_starts_at_a_typical_speaking_pitch() -> None:
    mean, log_spread = F0RangePredictor(8, 16)(torch.randn(3, 8))

    assert torch.allclose(mean.exp(), torch.full((3,), 150.0))
    assert torch.allclose(log_spread.exp(), torch.full((3,), 0.2))


def _harmonics(mode: str = "comb", bands: int | None = 8) -> HarmonicConditioning:
    model = tiny_decoder()
    assert model.harmonics is not None
    return HarmonicConditioning(mode, model.harmonics.mel_weights, 16_000, 320, 8, 8.0, bands)  # type: ignore[arg-type]


def test_the_comb_peaks_at_harmonics_and_is_silent_when_unvoiced() -> None:
    harmonics = _harmonics()
    log_f0 = torch.full((1, 2), math.log(200.0))
    voiced = torch.tensor([[1.0, 0.0]])

    features = harmonics(torch.zeros(1, 2), voiced, log_f0, None)

    assert features.shape == (1, harmonics.channels, 2)
    comb = features[0, 3:3 + 80]
    linear = harmonics.comb(log_f0)[0, 0]
    centers = (harmonics.mel_weights * harmonics.frequencies[:, None]).sum(0)
    near_harmonic = (centers / 200.0 - (centers / 200.0).round()).abs() < 0.1
    low = centers < 1500
    assert linear[low & near_harmonic].mean() > 2 * linear[low & ~near_harmonic].mean()
    assert torch.equal(comb[:, 1], torch.zeros(80))


def test_sinusoid_mode_and_missing_aperiodicity() -> None:
    harmonics = _harmonics("sinusoid")
    features = harmonics(torch.zeros(2, 5), torch.ones(2, 5), torch.full((2, 5), math.log(120.0)), None)

    assert harmonics.channels == 3 + 2 * 8 + 8
    assert features.shape == (2, harmonics.channels, 5)
    assert torch.equal(features[:, -8:], torch.zeros(2, 8, 5))
    assert features[:, 3:19].abs().max() <= 1.0


def _prosody_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    frames = batch["waveforms"].shape[1] // 320 + 1
    batch["f0"] = torch.cat([_f0(frames), 1.3 * _f0(frames)])
    batch["aperiodicity"] = torch.full((2, frames, 8), 0.3)
    return batch


@pytest.mark.parametrize("mode", ["comb", "sinusoid"])
def test_decoding_with_f0_trains_the_range_head(mode: str) -> None:
    model = tiny_decoder(harmonic_mode=mode, f0_dropout_rate=0.0, cfg_dropout_rate=0.0).train()
    batch = _prosody_batch()

    output = model(**batch)

    assert output.f0_range is not None and output.f0_range.shape == (2, 2)
    assert output.f0_range_targets is not None and output.f0_range_mask is not None
    assert output.f0_range_mask.all()
    loss = output.f0_range.sum() + output.logits.sum()
    loss.backward()
    assert model.f0_range is not None
    assert all(p.grad is not None for p in model.f0_range.parameters())


def test_decoding_without_f0_has_no_range_term_and_runs() -> None:
    model = tiny_decoder().train()

    output = model(**_batch())

    assert output.f0_range is None
    assert torch.isfinite(output.logits).all()


def test_true_f0_only_changes_the_conditioning_through_the_range() -> None:
    model = tiny_decoder(f0_dropout_rate=0.0).eval()
    batch = _prosody_batch()
    frames = model.unit_encoder(batch["unit_ids"], batch["unit_id_lengths"]).shape[-1]
    mask = torch.ones(2, frames, dtype=torch.bool)

    stats = F0Statistics.from_f0(model.fit_frames(batch["f0"], frames), mask)
    rebuilt, _ = model.prosody(batch["f0"], batch["aperiodicity"], mask, stats.log_f0_mean, stats.log_f0_spread)
    true, _ = model.prosody(batch["f0"], batch["aperiodicity"], mask, use_true_f0=True)

    assert rebuilt is not None and true is not None
    assert torch.allclose(rebuilt, true, atol=1e-3)


def test_f0_dropout_zeroes_the_conditioning_for_dropped_rows() -> None:
    model = tiny_decoder(f0_dropout_rate=1.0, cfg_dropout_rate=0.0).train()
    seen: list[torch.Tensor] = []
    backbone = model.backbone.forward
    model.backbone.forward = lambda *a: seen.append(a[6]) or backbone(*a)  # type: ignore[method-assign]

    model(**_prosody_batch())

    assert torch.equal(seen[0], torch.zeros_like(seen[0]))


def test_flat_generation_uses_the_requested_pitch() -> None:
    model = tiny_decoder().eval()
    batch = _prosody_batch()
    flat = torch.where(batch["f0"] > 0, torch.full_like(batch["f0"], 165.0), torch.zeros_like(batch["f0"]))
    seen: list[torch.Tensor] = []
    backbone = model.backbone.forward
    model.backbone.forward = lambda *a: seen.append(a[6]) or backbone(*a)  # type: ignore[method-assign]

    out = model.generate(
        batch["unit_ids"], batch["unit_id_lengths"], batch["speaker_ids"], steps=2,
        f0=flat, aperiodicity=batch["aperiodicity"], log_f0_mean=torch.full((2,), math.log(165.0)),
    )

    prosody = seen[0]
    voiced = prosody[:, 1] > 0
    assert torch.equal(prosody[:, 0][voiced], torch.zeros(int(voiced.sum())))
    assert torch.allclose(prosody[:, 2][voiced], torch.full((int(voiced.sum()),), math.log(165.0 / 150.0)), atol=1e-4)
    assert torch.isfinite(out.mels).all()


def test_prosody_can_be_turned_off() -> None:
    model = tiny_decoder(prosody_conditioning=False).train()

    output = model(**_prosody_batch())

    assert model.harmonics is None and model.f0_range is None and output.f0_range is None
