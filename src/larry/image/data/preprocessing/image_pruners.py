from larry.common.data.preprocessing.pruners import Pruner
from larry.image.config.preprocessing.image_pruner_configs import ImagePrunerConfig


class ImagePruner[C: ImagePrunerConfig = ImagePrunerConfig](Pruner[C]):
    """Superclass for image pruners."""

    ...
