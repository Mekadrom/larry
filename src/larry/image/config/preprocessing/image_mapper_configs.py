import dataclasses

from torchvision.transforms import InterpolationMode

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig, SingleColumnMapperConfig, UrlMapperConfig


@dataclasses.dataclass(kw_only=True)
class ImageMapperConfig(SingleColumnMapperConfig):
    """Superclass for all image mapper configs."""

    input_column: str = "image"
    output_column: str = "image"


@dataclasses.dataclass(kw_only=True)
class UrlImageMapperConfig(UrlMapperConfig, ImageMapperConfig):
    """Config class for ``UrlImageMapper`` that overrides ``UrlMapperConfig`` fields with sensible defaults."""

    input_column: str = "url"
    output_column: str = "image"

    download_cache_dir: str = "/tmp/larry/image_url_mapper"


@dataclasses.dataclass(kw_only=True)
class ImageResizingMapperConfig(ImageMapperConfig):
    """Config for the ``ImageResizingMapper`` that sets standard defaults for the larry models concerned with images."""

    width: int = 224
    height: int = 224
    pad_to_equal: bool = True
    resize: bool = True
    resize_interpolation_mode: InterpolationMode = InterpolationMode.BILINEAR
    center_crop: bool = True


@dataclasses.dataclass(kw_only=True)
class LarryImageVAELatentExtractingMapperConfig(MapperConfig):
    vae_checkpoint_path: str
    image_column: str = "image"
    latents_output_column: str = "latents"
    shape_output_column: str = "latent_shape"
    image_size: int = 256
    size_multiple: int = 16

    batch_size: int = 32
    writer_batch_size: int = 32
    num_proc: int = 1

    def required_input_columns(self) -> list[str]:
        return [self.image_column]

    def required_output_columns(self) -> list[str]:
        return [self.latents_output_column, self.shape_output_column]
