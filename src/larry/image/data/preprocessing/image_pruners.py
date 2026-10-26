import io
from typing import Any

from PIL import Image

from larry.common.data.preprocessing.pruners import Pruner
from larry.image.config.preprocessing.image_pruner_configs import ImageAspectRatioPrunerConfig, ImagePrunerConfig


class ImagePruner[C: ImagePrunerConfig = ImagePrunerConfig](Pruner[C]):
    """Superclass for image pruners."""

    ...


class ImageAspectRatioPruner(ImagePruner[ImageAspectRatioPrunerConfig]):
    @staticmethod
    def size(image: Any) -> tuple[int, int]:
        if isinstance(image, Image.Image):
            return image.size

        with Image.open(io.BytesIO(image["bytes"])) as opened:
            return opened.size

    def keep_example(self, example: dict[str, Any]) -> bool:
        for image in example.values():
            width, height = self.size(image)
            if max(width, height) > self.config.max_aspect_ratio * max(1, min(width, height)):
                return False

        return True
