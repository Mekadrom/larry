from abc import ABC

from larry.common.training.trainers import LarryTrainer
from larry.image.config.training.image_trainer_configs import ImageLarryTrainerConfig
from larry.image.model.image_models import ImageModel


class ImageTrainer[M: ImageModel, C: ImageLarryTrainerConfig = ImageLarryTrainerConfig](LarryTrainer[M, C], ABC):
    ...
