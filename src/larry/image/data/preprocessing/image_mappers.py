import io
import time

import datasets
import torch
from PIL import Image
from datasets import Dataset, Features, DatasetDict
from datasets.features.features import FeatureType
from torchvision import transforms
from torchvision.io import encode_jpeg, decode_image, ImageReadMode

from larry.common.data.preprocessing.mappers import SingleColumnMapper, UrlMapper
from larry.common.data.utils import features_of
from larry.common.utils.types import EncodedImage
from larry.image.config.preprocessing.image_mapper_configs import UrlImageMapperConfig, ImageResizingMapperConfig, \
    ImageMapperConfig
from larry.image.data.preprocessing import image_transforms as larry_image_transforms


class ImageMapper[I, O, C: ImageMapperConfig = ImageMapperConfig](SingleColumnMapper[I, O, C]):
    def output_feature(self) -> FeatureType | None:
        return datasets.Image(decode=False)


class UrlImageMapper(
    UrlMapper[EncodedImage, UrlImageMapperConfig],
    ImageMapper[str, EncodedImage | None, UrlImageMapperConfig]
):
    def user_agent(self) -> str:
        return "larry-image-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def validate_content(self, example: str, content: bytes) -> None:
        Image.open(io.BytesIO(content)).verify()

    def extract_content(self, example: str, content: bytes) -> EncodedImage:
        return EncodedImage(bytes=content, path=example)


class ImageResizingMapper(ImageMapper[EncodedImage | None, EncodedImage | None, ImageResizingMapperConfig]):
    def __init__(self, config: ImageResizingMapperConfig) -> None:
        super().__init__(config)

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

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        features = super().preprocessing_features(dataset)
        if features is None:
            features = features_of(dataset).copy()
        features.update({
            self.config.output_column: datasets.Image(decode=False),
        })
        return features

    def preprocess_example(self, example: EncodedImage | None) -> EncodedImage | None:
        if example is None:
            return example
        image = decode_image(torch.frombuffer(bytearray(example["bytes"]), dtype=torch.uint8), mode=ImageReadMode.RGB)
        encoded = encode_jpeg(self.transform(image), quality=95)
        assert isinstance(encoded, torch.Tensor)
        return EncodedImage(bytes=encoded.cpu().numpy().tobytes(), path=example["path"])
