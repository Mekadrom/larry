import dataclasses

from larry.common.config.config import Config
from larry.common.config.model.model_configs import ModelConfig


@dataclasses.dataclass(kw_only=True)
class ImageModelConfig(ModelConfig):
    """Superclass config for all image model configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class PatchDiscriminatorConfig(Config):
    base_channels: int = 64
    n_layers: int = 3
    norm_groups: int = 32


@dataclasses.dataclass(kw_only=True)
class ImageVAEConfig(ImageModelConfig):
    in_channels: int = 3
    base_channels: int = 128
    channel_mults: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 4])
    n_res_blocks: int = 2
    latent_channels: int = 16
    mid_attention: bool = True
    norm_groups: int = 32
    init_std: float = 0.02
    discriminator: PatchDiscriminatorConfig = dataclasses.field(default_factory=PatchDiscriminatorConfig)

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.channel_mults) - 1)
