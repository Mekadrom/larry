from larry.common.data.preprocessing.pruners import Pruner
from larry.image.config.image_pruner_configs import ImagePrunerConfig


class ImagePruner[I, C=ImagePrunerConfig](Pruner[I, C]):
    ...
