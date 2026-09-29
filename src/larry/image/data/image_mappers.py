import io

import datasets
import requests
import torch
from PIL import Image
from datasets import Dataset, Features
from datasets.features.features import FeatureType
from torchvision import transforms
from torchvision.io import encode_jpeg, decode_image, ImageReadMode

from larry.common.data.preprocessing.mappers import Mapper
from larry.image.config.image_mapper_configs import UrlImageMapperConfig, ImageResizingMapperConfig, \
    ImageMapperConfig
from larry.image.data import image_transforms as larry_image_transforms
from larry.utils.types import EncodedImage


class ImageMapper[I, O, C=ImageMapperConfig](Mapper[I, O]):
    def output_feature(self) -> FeatureType | None:
        return datasets.Image(decode=False)


class UrlImageMapper(ImageMapper[str, EncodedImage | None, UrlImageMapperConfig]):
    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        features = super().preprocessing_features(dataset)
        features.update({
            self.config.output_column: datasets.Image(decode=False),
        })
        return features

    def preprocess_example(self, example: str) -> EncodedImage | None:
        response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = None
                response = requests.get(
                    example,
                    timeout=self.config.timeout,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ImagePreprocessor/1.0)"},
                )
                response.raise_for_status()
                content = response.content
                # test for corrupt files
                Image.open(io.BytesIO(content)).verify()
                return EncodedImage(bytes=content, path=example)
            except requests.exceptions.RequestException:
                if response is not None:
                    self.log.critical(
                        f"Failure to download image: {response.status_code} - {response.content[:200]} for url: {example}")
                else:
                    self.log.exception(f"Failure to download image:")
                if attempt == self.config.max_retries:
                    return None
            except Exception:
                self.log.exception(f"Failure to decode image:")
                return None
        return None


class ImageResizingMapper(ImageMapper[EncodedImage, EncodedImage | None, ImageResizingMapperConfig]):
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

    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        features = super().preprocessing_features(dataset)
        features.update({
            self.config.output_column: datasets.Image(decode=False),
        })
        return features

    def preprocess_example(self, example: EncodedImage) -> EncodedImage | None:
        if example is None:
            return example
        image = decode_image(torch.frombuffer(bytearray(example["bytes"]), dtype=torch.uint8), mode=ImageReadMode.RGB)
        return EncodedImage(
            bytes=encode_jpeg(self.transform(image), quality=95).numpy().tobytes(),
            path=example["path"]
        )
