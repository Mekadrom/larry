import dataclasses
from typing import Any, Literal

from larry.common.config.config import Config
from larry.common.config.data.data_configs import DatasetMixtureConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.image.config.criterion.image_criterion_configs import ImageVAECriterionConfig
from larry.image.config.data.image_data_configs import ImageMixtureConfig


@dataclasses.dataclass(kw_only=True)
class ImageLarryTrainerConfig[M: DatasetMixtureConfig[Any] = DatasetMixtureConfig](LarryTrainerConfig[M]):
    """Superclass for trainers for image models."""

    ...


@dataclasses.dataclass(kw_only=True)
class ImageVAEVizConfig(Config):
    examples_per_source: int = 4


@dataclasses.dataclass(kw_only=True)
class ImageVAETrainerConfig(ImageLarryTrainerConfig[ImageMixtureConfig]):
    optimizer_name: Literal["muon", "adamw"] = "adamw"
    criterion: ImageVAECriterionConfig = dataclasses.field(default_factory=ImageVAECriterionConfig)
    viz: ImageVAEVizConfig = dataclasses.field(default_factory=ImageVAEVizConfig)
