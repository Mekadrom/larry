import io

import datasets
import torch
from PIL import Image
from datasets.features.features import FeatureType
from torchvision import transforms
from torchvision.io import encode_jpeg, decode_image, ImageReadMode

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import UrlMapper, SingleColumnMapper
from larry.common.utils.types import EncodedImage
from larry.image.config.preprocessing.image_mapper_configs import UrlImageMapperConfig, ImageResizingMapperConfig
from larry.image.data.preprocessing import image_transforms as larry_image_transforms


class UrlImageMapper(
    UrlMapper[EncodedImage, UrlImageMapperConfig],
    SingleColumnMapper[str, EncodedImage | None, UrlImageMapperConfig]
):
    def user_agent(self) -> str:
        return "larry-image-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def validate_content(self, example: str, content: bytes) -> None:
        Image.open(io.BytesIO(content)).verify()

    def extract_content(self, example: str, content: bytes) -> EncodedImage:
        return EncodedImage(bytes=content, path=example)

    def output_feature(self) -> FeatureType | None:
        return datasets.Image(decode=False)


class ImageResizingMapper(SingleColumnMapper[EncodedImage | None, EncodedImage | None, ImageResizingMapperConfig]):
    def __init__(self, provenance: Provenance, config: ImageResizingMapperConfig) -> None:
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
        if example is None:
            return example
        image = decode_image(torch.frombuffer(bytearray(example["bytes"]), dtype=torch.uint8), mode=ImageReadMode.RGB)
        encoded = encode_jpeg(self.transform(image), quality=95)
        assert isinstance(encoded, torch.Tensor)
        return EncodedImage(bytes=encoded.cpu().numpy().tobytes(), path=example["path"])

    def output_feature(self) -> FeatureType | None:
        return datasets.Image(decode=False)
