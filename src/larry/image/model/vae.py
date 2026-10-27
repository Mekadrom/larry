import dataclasses

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.func import functional_call

from larry.common.model.init import WeightInitializer
from larry.common.model.model import LarryModelOutput
from larry.image.config.model.image_model_configs import ImageVAEConfig, PatchDiscriminatorConfig
from larry.image.model.image_models import ImageModel


@dataclasses.dataclass(kw_only=True)
class SpatialTransform:
    rotations: int = 0
    scale: float = 1.0

    def apply(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        x = torch.rot90(x, self.rotations, dims=(-2, -1))
        if size is None or tuple(x.shape[-2:]) == size:
            return x

        return F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)

    def latent_size(self, latents: torch.Tensor) -> tuple[int, int]:
        height, width = latents.shape[-2:]
        if self.rotations % 2:
            height, width = width, height

        return max(1, round(height * self.scale)), max(1, round(width * self.scale))


@dataclasses.dataclass(kw_only=True)
class ImageVAEOutput(LarryModelOutput):
    mean: torch.Tensor
    logvar: torch.Tensor
    latents: torch.Tensor
    equivariant_reconstructions: torch.Tensor
    equivariant_targets: torch.Tensor
    real_logits: torch.Tensor
    fake_logits: torch.Tensor
    generator_logits: torch.Tensor

    @property
    def reconstructions(self) -> torch.Tensor:
        return self.logits


def group_norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, channels), channels, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm = group_norm(channels, groups)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).transpose(1, 2).unsqueeze(1).chunk(3, dim=-1)
        h = F.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2).contiguous()
        return x + self.out(h.view(batch_size, channels, height, width))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class MidBlock(nn.Module):
    def __init__(self, channels: int, groups: int, attention: bool) -> None:
        super().__init__()
        self.res1 = ResBlock(channels, channels, groups)
        self.attention = SpatialAttention(channels, groups) if attention else nn.Identity()
        self.res2 = ResBlock(channels, channels, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.attention(self.res1(x)))


class VAEEncoder(nn.Module):
    def __init__(self, config: ImageVAEConfig) -> None:
        super().__init__()
        channels = [config.base_channels * m for m in config.channel_mults]
        self.conv_in = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        blocks: list[nn.Module] = []
        previous = channels[0]
        for level, width in enumerate(channels):
            for _ in range(config.n_res_blocks):
                blocks.append(ResBlock(previous, width, config.norm_groups))
                previous = width

            if level < len(channels) - 1:
                blocks.append(Downsample(width))

        self.blocks = nn.Sequential(*blocks)
        self.mid = MidBlock(previous, config.norm_groups, config.mid_attention)
        self.norm_out = group_norm(previous, config.norm_groups)
        self.conv_out = nn.Conv2d(previous, 2 * config.latent_channels, 3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.blocks(self.conv_in(images)))
        return self.conv_out(F.silu(self.norm_out(h)))


class VAEDecoder(nn.Module):
    def __init__(self, config: ImageVAEConfig) -> None:
        super().__init__()
        channels = [config.base_channels * m for m in reversed(config.channel_mults)]
        self.conv_in = nn.Conv2d(config.latent_channels, channels[0], 3, padding=1)
        self.mid = MidBlock(channels[0], config.norm_groups, config.mid_attention)
        blocks: list[nn.Module] = []
        previous = channels[0]
        for level, width in enumerate(channels):
            for _ in range(config.n_res_blocks + 1):
                blocks.append(ResBlock(previous, width, config.norm_groups))
                previous = width

            if level < len(channels) - 1:
                blocks.append(Upsample(width))

        self.blocks = nn.Sequential(*blocks)
        self.norm_out = group_norm(previous, config.norm_groups)
        self.conv_out = nn.Conv2d(previous, config.in_channels, 3, padding=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.mid(self.conv_in(latents)))
        return self.conv_out(F.silu(self.norm_out(h)))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, config: PatchDiscriminatorConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, config.base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        ]

        channels = config.base_channels
        for n in range(1, config.n_layers + 1):
            out_channels = config.base_channels * min(2 ** n, 8)
            layers += [
                nn.Conv2d(channels, out_channels, 4, stride=2 if n < config.n_layers else 1, padding=1),
                group_norm(out_channels, config.norm_groups),
                nn.LeakyReLU(0.2),
            ]
            channels = out_channels

        layers.append(nn.Conv2d(channels, 1, 4, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class ImageVAE(ImageModel[ImageVAEConfig], PyTorchModelHubMixin):
    def __init__(self, config: ImageVAEConfig) -> None:
        super().__init__(config)
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)
        self.discriminator = PatchDiscriminator(config.in_channels, config.discriminator)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        WeightInitializer("normal", self.config.init_std).apply(self)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(images).chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def frozen_discriminator(self, images: torch.Tensor) -> torch.Tensor:
        parameters = {name: p.detach() for name, p in self.discriminator.named_parameters()}
        return functional_call(self.discriminator, parameters, (images,))

    def forward(
            self,
            images: torch.Tensor,
            transform: SpatialTransform,
            equivariance_gradients: bool = True,
            adversarial_gradients: bool = True,
    ) -> ImageVAEOutput:
        mean, logvar = self.encode(images)
        latents = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        reconstructions = self.decode(latents)

        size = transform.latent_size(latents)
        factor = self.config.downsample_factor
        with torch.set_grad_enabled(equivariance_gradients and torch.is_grad_enabled()):
            equivariant_reconstructions = self.decode(transform.apply(latents, size))

        equivariant_targets = transform.apply(images, (size[0] * factor, size[1] * factor))

        with torch.set_grad_enabled(adversarial_gradients and torch.is_grad_enabled()):
            generator_logits = self.frozen_discriminator(reconstructions)

        return ImageVAEOutput(
            logits=reconstructions,
            mean=mean,
            logvar=logvar,
            latents=latents,
            equivariant_reconstructions=equivariant_reconstructions,
            equivariant_targets=equivariant_targets,
            real_logits=self.discriminator(images),
            fake_logits=self.discriminator(reconstructions.detach()),
            generator_logits=generator_logits,
        )
