from abc import ABC
from typing import Any

from larry.common.config.data.data_configs import DatasetSourceConfig
from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.training.trainers import LarryTrainer
from larry.image.config.training.image_trainer_configs import ImageLarryTrainerConfig
from larry.image.model.image_models import ImageModel


class ImageTrainer[
M: ImageModel,
C: ImageLarryTrainerConfig[Any] = ImageLarryTrainerConfig[Any],
S: DatasetSourceConfig = DatasetSourceConfig,
D: IndexedDataset[Any] = IndexedDataset[Any],
](LarryTrainer[M, C, S, D], ABC):
    """Superclass for all image trainers."""

    ...
