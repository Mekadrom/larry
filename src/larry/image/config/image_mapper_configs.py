import dataclasses

from torchvision.transforms import InterpolationMode

from larry.common.config.mapper_configs import MapperConfig


@dataclasses.dataclass(kw_only=True)
class ImageMapperConfig(MapperConfig):
    """Super class for all DTOs related to image dataset preprocessing."""
    input_column: str = "image"
    output_column: str | None = "image"


@dataclasses.dataclass(kw_only=True)
class UrlImageMapperConfig(ImageMapperConfig):
    input_column: str = "url"

    timeout: float = 1.0
    max_retries: int = 2


@dataclasses.dataclass(kw_only=True)
class ImageResizingMapperConfig(ImageMapperConfig):
    width: int = 224
    height: int = 224
    pad_to_equal: bool = True
    resize: bool = True
    resize_interpolation_mode: InterpolationMode | str = InterpolationMode.BILINEAR
    center_crop: bool = True
