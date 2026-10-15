import dataclasses

from torchvision.transforms import InterpolationMode

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig


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
