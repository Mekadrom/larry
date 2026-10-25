import dataclasses

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig


@dataclasses.dataclass(kw_only=True)
class ImageSourceConfig(DatasetSourceConfig):
    image_column: str = "image"


@dataclasses.dataclass(kw_only=True)
class ImageMixtureConfig(DatasetMixtureConfig[ImageSourceConfig]):
    # the shorter side is resized to this, then cropped square
    crop_size: int = 256
    random_flip: bool = True
