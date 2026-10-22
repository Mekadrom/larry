import pytest
import torch

from larry.common.config.model.model_configs import LarryConditionalEncoderBlockModuleConfig
from larry.common.model.llm import LarryConditionalEncoderBlock
from larry.common.model.norms import AdaLayerNormZero
from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig
from larry.voice.config.model.voice_model_configs import SpeechDecoderModelConfig
from larry.voice.criteria.flow_matching import SpeechDecoderCriterion
from larry.voice.model.speech_decoder import SpeechDecoderModel


def tiny_config(**overrides: object) -> SpeechDecoderModelConfig:
    return SpeechDecoderModelConfig(**{
        "n_speakers": 3,
        "d_model": 64,
        "n_heads": 4,
        "n_query_groups": 4,
        "rotary_embedding_dim": 16,
        "speaker_dim": 8,
        "n_encoder_layers": 1,
        "n_backbone_layers": 2,
        "fsq_dims": 4,
        "mel_mean": -2.0,
        "mel_std": 1.5,
        **overrides,
    })


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "unit_ids": torch.randint(0, 81, (2, 25), generator=generator),
        "unit_id_lengths": torch.tensor([25, 18]),
        "speaker_ids": torch.tensor([0, 2]),
        "waveforms": torch.randn(2, 16_000, generator=generator) * 0.1,
        "waveform_lengths": torch.tensor([16_000, 11_000]),
    }


def _model(**overrides: object) -> SpeechDecoderModel:
    torch.manual_seed(0)
    return SpeechDecoderModel(tiny_config(**overrides))


def test_gated_norm_starts_closed() -> None:
    norm = AdaLayerNormZero(16, 8)

    _, gate = norm(torch.randn(2, 5, 16), torch.randn(2, 8))

    assert not gate.any()


def test_conditional_block_starts_as_the_identity() -> None:
    torch.manual_seed(0)
    block = LarryConditionalEncoderBlock(LarryConditionalEncoderBlockModuleConfig(
        d_model=32, n_heads=4, n_query_groups=4, rotary_embedding_dim=8
    ))
    x = torch.randn(2, 7, 32)

    assert torch.equal(block(x, torch.randn(2, 32), torch.ones(2, 7)).logits, x)


def test_conditional_block_is_not_causal_by_default() -> None:
    assert not LarryConditionalEncoderBlockModuleConfig().is_causal
    assert not tiny_config().is_causal


def test_forward_crops_targets_and_conditioning_to_a_common_width() -> None:
    model = _model()
    batch = _batch()

    output = model(**batch)

    mels, mel_lengths = model.frontend(batch["waveforms"], batch["waveform_lengths"])
    frames = min(mels.shape[-1], 4 * batch["unit_ids"].shape[1])
    assert output.logits.shape == output.targets.shape == (2, 80, frames)
    expected = torch.minimum(mel_lengths, 4 * batch["unit_id_lengths"]).clamp(max=frames)
    assert output.mask.sum(dim=1).tolist() == expected.tolist()
    assert not output.logits[1, :, int(expected[1]):].any()


def test_mels_normalize_and_denormalize_exactly() -> None:
    model = _model()
    mels = torch.randn(2, 80, 9) * 2 - 2

    assert torch.allclose(model.denormalize(model.normalize(mels)), mels, atol=1e-6)


def test_a_training_step_gives_a_finite_loss_and_gradients() -> None:
    model = _model().train()
    criterion = SpeechDecoderCriterion(SpeechDecoderCriterionConfig())

    loss = criterion(model(**_batch()), step=500)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.backbone.out.weight.grad is not None and model.backbone.out.weight.grad.abs().sum() > 0
    assert model.speaker_embedding.mean_embedding.weight.grad is not None


@pytest.mark.parametrize(("training", "rate", "dropped"), [(True, 1.0, True), (False, 1.0, False), (True, 0.0, False)])
def test_cfg_dropout_only_applies_while_training(training: bool, rate: float, dropped: bool) -> None:
    model = _model(cfg_dropout_rate=rate)
    model.train(training)

    output = model(**_batch())

    null = model.cfg_null_speaker_embedding.expand_as(output.speaker_embeddings)
    assert torch.equal(output.speaker_embeddings, null) is dropped


def test_cfg_dropout_rate_matches_the_config() -> None:
    model = _model(cfg_dropout_rate=0.1).train()
    model.cfg_null_speaker_embedding.data.fill_(123.0)
    batch = {k: v.repeat(500, *([1] * (v.dim() - 1))) for k, v in _batch().items()}

    with torch.no_grad():
        speaker_embs = model(**batch).speaker_embeddings

    assert (speaker_embs == 123.0).all(dim=1).float().mean() == pytest.approx(0.1, abs=0.03)


@pytest.mark.parametrize("cfg_scale", [1.0, 2.0])
def test_generation_covers_every_unit_and_masks_padding(cfg_scale: float) -> None:
    model = _model().eval()
    batch = _batch()

    out = model.generate(
        batch["unit_ids"], batch["unit_id_lengths"], batch["speaker_ids"], steps=2, cfg_scale=cfg_scale,
        generator=torch.Generator().manual_seed(0),
    )

    assert out.mels.shape == (2, 80, 100)
    assert out.mel_lengths.tolist() == [100, 72]
    assert not out.mels[1, :, 72:].any()
    assert torch.isfinite(out.mels).all()


def test_generation_is_seeded() -> None:
    model = _model().eval()
    batch = _batch()

    def run() -> torch.Tensor:
        return model.generate(
            batch["unit_ids"], batch["unit_id_lengths"], batch["speaker_ids"], steps=2, cfg_scale=2.0,
            generator=torch.Generator().manual_seed(4),
        ).mels

    assert torch.equal(run(), run())


@pytest.mark.parametrize(("step", "scale"), [(None, 1.0), (0, 0.0), (5_000, 0.5), (10_000, 1.0), (50_000, 1.0)])
def test_kl_weight_warms_up(step: int | None, scale: float) -> None:
    criterion = SpeechDecoderCriterion(SpeechDecoderCriterionConfig(kl_weight=0.2, kl_warmup_steps=10_000))

    assert criterion.kl_scale(step) == pytest.approx(0.2 * scale)


def test_criterion_adds_the_scaled_kl_to_the_flow_loss() -> None:
    model = _model().train()
    criterion = SpeechDecoderCriterion(SpeechDecoderCriterionConfig(kl_weight=0.5, kl_warmup_steps=0))
    output = model(**_batch())

    total = criterion(output, step=3)

    expected = criterion.flow_loss(output) + criterion.prior_loss(output) + 0.5 * output.kl.mean()
    assert float(total) == pytest.approx(float(expected), rel=1e-5)


def test_prior_loss_only_counts_valid_frames() -> None:
    model = _model().train()
    criterion = SpeechDecoderCriterion(SpeechDecoderCriterionConfig())
    output = model(**_batch())
    before = criterion.prior_loss(output)

    output.prior[1, :, int(output.frame_mask[1].sum()):] += 100.0

    assert torch.equal(criterion.prior_loss(output), before)


def test_infill_is_off_by_default_and_keeps_the_two_input_architecture() -> None:
    model = _model()

    assert not model.config.infill_enabled
    assert model.backbone.embed.in_channels == 2 * 80


def test_infill_hides_one_span_and_only_trains_on_it() -> None:
    model = _model(infill_enabled=True, cfg_dropout_rate=0.0).train()
    assert model.backbone.embed.in_channels == 3 * 80

    output = model(**_batch())

    hidden = output.mask.bool()
    valid = output.frame_mask.bool()
    assert (hidden <= valid).all()
    for row in range(2):
        fraction = float(hidden[row].sum() / valid[row].sum())
        assert 0.69 <= fraction <= 1.0
        span = hidden[row].nonzero().flatten()
        assert (span.diff() == 1).all()


def test_infill_context_never_reveals_the_hidden_span() -> None:
    model = _model(infill_enabled=True, cfg_dropout_rate=0.0).train()
    seen: dict[str, torch.Tensor] = {}
    forward = model.backbone.forward

    def capture(*args: torch.Tensor) -> torch.Tensor:
        seen["context"] = args[5]
        return forward(*args)

    model.backbone.forward = capture
    output = model(**_batch())

    assert not (seen["context"] * output.mask.unsqueeze(1)).any()
    assert (seen["context"] * (output.frame_mask - output.mask).unsqueeze(1)).abs().sum() > 0


def test_cfg_dropout_also_drops_the_infill_context() -> None:
    model = _model(infill_enabled=True, cfg_dropout_rate=1.0).train()
    seen: dict[str, torch.Tensor] = {}
    forward = model.backbone.forward

    def capture(*args: torch.Tensor) -> torch.Tensor:
        seen["context"] = args[5]
        return forward(*args)

    model.backbone.forward = capture
    model(**_batch())

    assert not seen["context"].any()


def test_generation_keeps_the_prompt_frames_and_fills_the_rest() -> None:
    model = _model(infill_enabled=True).eval()
    batch = _batch()
    prompt = torch.randn(2, 80, 30) - 2
    prompt_lengths = torch.tensor([30, 20])

    out = model.generate(
        batch["unit_ids"], batch["unit_id_lengths"], None, steps=2, cfg_scale=2.0,
        generator=torch.Generator().manual_seed(0), prompt_mels=prompt, prompt_mel_lengths=prompt_lengths,
    )

    assert torch.allclose(out.mels[0, :, :30], prompt[0], atol=1e-5)
    assert torch.allclose(out.mels[1, :, :20], prompt[1, :, :20], atol=1e-5)
    assert torch.isfinite(out.mels).all()


def test_a_prompt_requires_an_infill_model() -> None:
    batch = _batch()

    with pytest.raises(ValueError, match="infill_enabled"):
        _model().generate(
            batch["unit_ids"], batch["unit_id_lengths"], batch["speaker_ids"], steps=1,
            prompt_mels=torch.zeros(2, 80, 4), prompt_mel_lengths=torch.tensor([4, 4]),
        )


def test_generation_without_speaker_ids_uses_the_null_speaker() -> None:
    model = _model().eval()
    batch = _batch()

    out = model.generate(batch["unit_ids"], batch["unit_id_lengths"], None, steps=2, sway_coefficient=-1.0)

    assert torch.isfinite(out.mels).all()
