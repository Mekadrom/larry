import io

import datasets
import torch
from datasets.features.features import FeatureType
from PIL import Image
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_image, encode_jpeg

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import SingleColumnMapper
from larry.common.data.preprocessing.url_mappers import UrlMapper
from larry.common.data.utils import bot_user_agent
from larry.common.utils.types import EncodedImage
from larry.image.config.preprocessing.image_mapper_configs import ImageResizingMapperConfig, UrlImageMapperConfig
from larry.image.data.preprocessing import image_transforms as larry_image_transforms


class UrlImageMapper(UrlMapper[EncodedImage, UrlImageMapperConfig]):
    """Implementation of ``UrlMapper`` meant for downloading images, maps to an undecoded image feature type column."""

    def user_agent(self) -> str:
        """Overrides the user agent to uniquely identify pipelines using this mapper from this project."""
        return bot_user_agent("larry-image-downloader")

    def validate_content(self, example: str, content: bytes) -> None:
        """Validates that the content defines a real image without actually decoding it. Will raise if not."""
        Image.open(io.BytesIO(content)).verify()

    def extract_content(self, example: str, content: bytes) -> EncodedImage:
        """Returns the wrapper type for images in preprocessing pipelines as (bytes, path)."""
        return EncodedImage(bytes=content, path=example)

    def output_feature(self) -> FeatureType | None:
        """Sets the output feature type of this mapper's output column to an unmaterialized raw image as bytes.

        Saves memory.
        """
        return datasets.Image(decode=False)


class ImageResizingMapper(SingleColumnMapper[EncodedImage | None, EncodedImage | None, ImageResizingMapperConfig]):
    """Resizes images, optionally cropping, padding, or centering (need to do at least one)."""

    def __init__(self, provenance: Provenance, config: ImageResizingMapperConfig) -> None:
        """Initializes the transforms that this mapper will use to resize images according to the config provided."""
        super().__init__(provenance, config)

        transform = []
        if self.config.pad_to_equal:
            transform.append(larry_image_transforms.PadToEqualSize())
        if self.config.resize:
            transform.append(
                transforms.Resize(
                    size=max(self.config.height, self.config.width),
                    interpolation=self.config.resize_interpolation_mode,
                )
            )
        if self.config.center_crop:
            transform.append(transforms.CenterCrop(size=(self.config.height, self.config.width)))

        self.transform = transforms.Compose(transform)

    def preprocess_example(self, example: EncodedImage | None) -> EncodedImage | None:
        """Performs the transform mapping, not operating on nulls."""
        if example is None:
            return None

        image = decode_image(torch.frombuffer(bytearray(example["bytes"]), dtype=torch.uint8), mode=ImageReadMode.RGB)
        encoded = encode_jpeg(self.transform(image), quality=95)
        assert isinstance(encoded, torch.Tensor)
        return EncodedImage(bytes=encoded.cpu().numpy().tobytes(), path=example["path"])

    def output_feature(self) -> FeatureType | None:
        """Sets the output feature for this mapper's output column to an undecoded image."""
        return datasets.Image(decode=False)
