from abc import ABC

from larry.common.training.training_model import LarryModel
from larry.image.config.model.image_model_configs import ImageModelConfig


class ImageModel[C: ImageModelConfig = ImageModelConfig](LarryModel[C], ABC):
    ...
