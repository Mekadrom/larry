import dataclasses

from torchvision.transforms import InterpolationMode

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, UrlMapperConfig


@dataclasses.dataclass(kw_only=True)
class ImageMapperConfig(SingleColumnMapperConfig):
    input_column: str = "image"
    output_column: str = "image"


@dataclasses.dataclass(kw_only=True)
class UrlImageMapperConfig(UrlMapperConfig, ImageMapperConfig):
    input_column: str = "url"
    output_column: str = "image"

    download_cache_dir: str = "/tmp/larry/image_url_mapper"

    hash_column: str = "image_url_hash"


@dataclasses.dataclass(kw_only=True)
class ImageResizingMapperConfig(ImageMapperConfig):
    width: int = 224
    height: int = 224
    pad_to_equal: bool = True
    resize: bool = True
    resize_interpolation_mode: InterpolationMode = InterpolationMode.BILINEAR
    center_crop: bool = True
