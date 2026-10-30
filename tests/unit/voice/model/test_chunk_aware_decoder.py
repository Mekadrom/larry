import pytest
import torch

from larry.voice.model.speech_decoder import chunk_attention_mask
from tests.unit.voice.model.test_speech_decoder import _batch
from tests.unit.voice.model.test_speech_decoder import _model as tiny_decoder

UPSAMPLE = 4


def test_chunk_masks_see_their_own_and_earlier_chunks_only() -> None:
    valid = torch.tensor([[True] * 5 + [False]])

    mask = chunk_attention_mask(valid, 2)[0, 0]

    assert mask[0].tolist() == [True, True, False, False, False, False]
    assert mask[3].tolist() == [True, True, True, True, False, False]
    assert mask[4].tolist() == [True, True, True, True, True, False]
    assert torch.equal(chunk_attention_mask(valid, None), valid)


def test_each_unit_upsamples_to_exactly_its_own_frames() -> None:
    model = tiny_decoder().eval()
    unit_ids, lengths = _batch()["unit_ids"][:1], torch.tensor([25])
    changed = unit_ids.clone()
    changed[0, 10] = (changed[0, 10] + 1) % 81

    with torch.no_grad():
        before = model.unit_encoder.upsampler(torch.zeros(1, 64, 25))
        assert before.shape[-1] == 25 * UPSAMPLE
        a = model.unit_encoder(unit_ids, lengths, chunk_units=1)
        b = model.unit_encoder(changed, lengths, chunk_units=1)

    differs = (a - b).abs().amax(dim=(0, 1)) > 0
    assert differs.nonzero().flatten().tolist() == list(range(10 * UPSAMPLE, 25 * UPSAMPLE))[:differs.sum()]
    assert not differs[: 10 * UPSAMPLE].any()


@pytest.mark.parametrize("chunk_units", [3, 6])
def test_later_units_never_change_earlier_unit_conditioning(chunk_units: int) -> None:
    model = tiny_decoder().eval()
    unit_ids, lengths = _batch()["unit_ids"][:1], torch.tensor([25])
    boundary = 2 * chunk_units
    changed = unit_ids.clone()
    changed[0, boundary:] = (changed[0, boundary:] + 5) % 81

    with torch.no_grad():
        a = model.unit_encoder(unit_ids, lengths, chunk_units)
        b = model.unit_encoder(changed, lengths, chunk_units)
        full_a = model.unit_encoder(unit_ids, lengths)
        full_b = model.unit_encoder(changed, lengths)

    assert torch.allclose(a[..., : boundary * UPSAMPLE], b[..., : boundary * UPSAMPLE], atol=1e-6)
    assert not torch.allclose(full_a[..., : boundary * UPSAMPLE], full_b[..., : boundary * UPSAMPLE], atol=1e-6)


def _active(model: torch.nn.Module) -> torch.nn.Module:
    torch.nn.init.normal_(model.backbone.out.weight, std=0.1)  # type: ignore[union-attr]
    return model


def test_the_velocity_network_never_looks_past_the_current_chunk() -> None:
    model = _active(tiny_decoder().eval())
    torch.manual_seed(1)
    frames, chunk_frames = 40, 8
    xt, unit_cond = torch.randn(1, 80, frames), torch.randn(1, 80, frames)
    assert model.harmonics is not None
    prosody = torch.randn(1, model.harmonics.channels, frames)
    t, speaker = torch.rand(1), torch.randn(1, 64)
    mask = torch.ones(1, frames)
    attention = chunk_attention_mask(mask > 0, chunk_frames)
    boundary = 2 * chunk_frames
    later = torch.zeros(1, 1, frames)
    later[..., boundary:] = 1.0

    with torch.no_grad():
        a = model.backbone(xt, t, speaker, unit_cond, mask, None, prosody, attention)
        b = model.backbone(xt + later, t, speaker, unit_cond + later, mask, None, prosody + later, attention)

    assert torch.allclose(a[..., :boundary], b[..., :boundary], atol=1e-5)
    assert not torch.allclose(a[..., boundary:], b[..., boundary:], atol=1e-5)


def test_chunked_generation_keeps_earlier_mels_when_later_units_change() -> None:
    model = _active(tiny_decoder().eval())
    batch = _batch()
    unit_ids, lengths, speakers = batch["unit_ids"][:1], torch.tensor([25]), batch["speaker_ids"][:1]
    changed = unit_ids.clone()
    changed[0, 12:] = (changed[0, 12:] + 7) % 81

    def generate(units: torch.Tensor) -> torch.Tensor:
        return model.generate(
            units, lengths, speakers, steps=3, cfg_scale=2.0, generator=torch.Generator().manual_seed(0), chunk_units=6,
        ).mels

    with torch.no_grad():
        a, b = generate(unit_ids), generate(changed)

    assert torch.allclose(a[..., : 12 * UPSAMPLE], b[..., : 12 * UPSAMPLE], atol=1e-5)
    assert not torch.allclose(a[..., 12 * UPSAMPLE:], b[..., 12 * UPSAMPLE:], atol=1e-5)


@pytest.mark.parametrize(("rate", "expected"), [(1.0, {None}), (0.0, {6, 12, 25})])
def test_training_mixes_full_context_and_chunk_sizes(rate: float, expected: set[int | None]) -> None:
    model = tiny_decoder(full_context_rate=rate)
    torch.manual_seed(0)

    seen = {model.sample_chunk_units() for _ in range(200)}

    assert seen == expected


def test_training_uses_chunk_masks_and_eval_uses_full_context() -> None:
    model = tiny_decoder(full_context_rate=0.0, cfg_dropout_rate=0.0, f0_dropout_rate=0.0)
    masks: list[torch.Tensor | None] = []
    backbone = model.backbone.forward
    model.backbone.forward = lambda *a: masks.append(a[7]) or backbone(*a)  # type: ignore[method-assign]

    model.train()(**_batch())
    model.eval()(**_batch())

    assert masks[0] is not None and masks[0].dim() == 4
    assert masks[1] is None
