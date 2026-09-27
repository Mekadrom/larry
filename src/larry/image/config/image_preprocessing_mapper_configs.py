import dataclasses

from torchvision.transforms import InterpolationMode

from larry.common.config.preprocessing_mapper_configs import PreprocessingMapperConfig


@dataclasses.dataclass(kw_only=True)
class ImagePreprocessingMapperConfig(PreprocessingMapperConfig):
    """Super class for all DTOs related to image dataset preprocessing."""
    input_column: str = "image"
    output_column: str | list[dict[str, str]] | None = "image"


@dataclasses.dataclass(kw_only=True)
class URLDownloadingImagePreprocessingMapperConfig(ImagePreprocessingMapperConfig):
    input_column: str = "url"

    timeout: float = 1.0
    max_retries: int = 2


@dataclasses.dataclass(kw_only=True)
class ImageResizingPreprocessingMapperConfig(ImagePreprocessingMapperConfig):
    width: int = 224
    height: int = 224
    pad_to_equal: bool = True
    resize: bool = True
    resize_interpolation_mode: InterpolationMode | str = InterpolationMode.BILINEAR
    center_crop: bool = True
    mean: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclasses.dataclass(kw_only=True)
class ImageVAELatentMapperConfig(ImagePreprocessingMapperConfig):
    """Config for tokenization of text."""

    vae_latent_encoder_model_config: str

    output_column: str | list[dict[str, str]] | None = "latents"
