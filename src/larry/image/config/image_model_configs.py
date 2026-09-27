import dataclasses

from larry.common.config.model_config import ModelConfig


@dataclasses.dataclass
class ImageModelConfig(ModelConfig):
    """Super class for all DTOs related to image model configuration."""


@dataclasses.dataclass
class ImageVAEModelConfig(ImageModelConfig):
    """Model config for configuring a VAE."""

    latent_shape: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ImageVAEEncoderModelConfig(ImageVAEModelConfig):
    """Model config for just the encoder of a VAE."""


@dataclasses.dataclass
class ImageVAEDecoderModelConfig(ImageVAEModelConfig):
    """Model config for just the decoder of a VAE."""
