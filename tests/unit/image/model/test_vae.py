from pathlib import Path

import pytest
import torch

from larry.image.config.criterion.image_criterion_configs import ImageVAECriterionConfig
from larry.image.config.model.image_model_configs import ImageVAEConfig, PatchDiscriminatorConfig
from larry.image.criteria.vae import ImageVAECriterion
from larry.image.model.vae import ImageVAE, SpatialTransform

TINY = ImageVAEConfig(
    base_channels=16, channel_mults=[1, 2, 2], n_res_blocks=1, latent_channels=4, norm_groups=4,
    discriminator=PatchDiscriminatorConfig(base_channels=8, n_layers=2, norm_groups=4),
)


def _images(height: int = 32, width: int = 24) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(2, 3, height, width) * 2 - 1


@pytest.mark.parametrize(("height", "width"), [(32, 24), (48, 64)])
def test_any_size_divisible_by_the_factor_round_trips(height: int, width: int) -> None:
    model = ImageVAE(TINY)

    output = model(_images(height, width), SpatialTransform())

    assert output.latents.shape == (2, 4, height // 4, width // 4)
    assert output.reconstructions.shape == (2, 3, height, width)


@pytest.mark.parametrize(("rotations", "scale"), [(1, 1.0), (2, 0.5), (3, 0.75)])
def test_equivariance_targets_match_the_transformed_decode(rotations: int, scale: float) -> None:
    model = ImageVAE(TINY)

    output = model(_images(32, 24), SpatialTransform(rotations=rotations, scale=scale))

    assert output.equivariant_reconstructions.shape == output.equivariant_targets.shape
    latent_height, latent_width = (6, 8) if rotations % 2 else (8, 6)
    expected = (round(latent_height * scale) * 4, round(latent_width * scale) * 4)
    assert tuple(output.equivariant_targets.shape[-2:]) == expected


def test_rotation_targets_are_exact_image_rotations() -> None:
    images = _images()

    assert torch.equal(SpatialTransform(rotations=1).apply(images, (24, 32)), torch.rot90(images, 1, dims=(-2, -1)))


def _terms(config: ImageVAECriterionConfig, step: int = 0) -> tuple[ImageVAE, dict]:
    model = ImageVAE(TINY)
    images = _images()
    criterion = ImageVAECriterion(config)
    output = model(
        images,
        SpatialTransform(rotations=1, scale=0.5),
        equivariance_gradients=config.equivariance_weight != 0,
        adversarial_gradients=criterion.adversarial_weight(step) != 0,
    )
    return model, criterion.loss_terms(output, images=images, step=step)


def test_every_term_is_logged_and_zero_weights_carry_no_gradient() -> None:
    _, terms = _terms(ImageVAECriterionConfig())

    assert set(terms) == {"reconstruction", "frequency", "kl", "equivariance", "adversarial", "discriminator"}
    assert all(torch.isfinite(term.value) for term in terms.values())
    for name in ("frequency", "equivariance", "adversarial"):
        assert terms[name].weight == 0 and not terms[name].sums.requires_grad, name


def test_the_adversarial_term_trains_the_decoder_but_never_the_discriminator() -> None:
    model, terms = _terms(ImageVAECriterionConfig(adversarial_weight=1.0))

    terms["adversarial"].value.backward()

    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.discriminator.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())


def test_the_discriminator_term_trains_only_the_discriminator() -> None:
    model, terms = _terms(ImageVAECriterionConfig())

    terms["discriminator"].value.backward()

    assert all(p.grad is None for p in model.decoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.discriminator.parameters())


def test_the_adversarial_weight_starts_at_the_configured_step() -> None:
    criterion = ImageVAECriterion(ImageVAECriterionConfig(adversarial_weight=0.5, adversarial_start_step=10))

    assert criterion.adversarial_weight(9) == 0.0
    assert criterion.adversarial_weight(10) == 0.5
    assert criterion.adversarial_weight(None) == 0.5


def test_equivariance_gradients_reach_the_encoder_when_weighted() -> None:
    model, terms = _terms(ImageVAECriterionConfig(equivariance_weight=1.0))

    terms["equivariance"].value.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_attention_weight_gradients_keep_the_parameter_layout() -> None:
    from larry.image.model.vae import SpatialAttention

    attention = SpatialAttention(16, 4)
    attention(torch.randn(2, 16, 4, 6)).sum().backward()

    for parameter in (attention.qkv.weight, attention.out.weight):
        assert parameter.grad is not None and parameter.grad.stride() == parameter.stride()


def test_a_saved_vae_loads_back_with_its_discriminator_config(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = ImageVAE(TINY)
    model.save_pretrained(tmp_path)

    loaded = ImageVAE.from_pretrained(str(tmp_path))

    assert loaded.config.discriminator == TINY.discriminator
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True))
