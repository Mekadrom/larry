import dataclasses

from torchvision.transforms import InterpolationMode

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig


@dataclasses.dataclass(kw_only=True)
class ImageSingleColumnMapperConfig(SingleColumnMapperConfig):
    input_column: str = "image"
    output_column: str = "image"


@dataclasses.dataclass(kw_only=True)
class UrlImageMapperConfig(ImageSingleColumnMapperConfig):
    input_column: str = "url"

    timeout: float = 1.0
    max_retries: int = 2


@dataclasses.dataclass(kw_only=True)
class ImageResizingMapperConfig(ImageSingleColumnMapperConfig):
    width: int = 224
    height: int = 224
    pad_to_equal: bool = True
    resize: bool = True
    resize_interpolation_mode: InterpolationMode = InterpolationMode.BILINEAR
    center_crop: bool = True
