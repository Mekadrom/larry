from larry.common.data.preprocessing.pruners import Pruner
from larry.image.config.preprocessing.image_pruner_configs import ImagePrunerConfig


class ImagePruner[I, C: ImagePrunerConfig = ImagePrunerConfig](Pruner[I, C]):
    ...
