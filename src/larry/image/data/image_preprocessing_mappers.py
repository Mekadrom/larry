import io
from typing import Any, Sequence

import datasets
import requests
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from datasets import Dataset, Features
from torchvision import transforms
from torchvision.io import encode_jpeg

from larry.common.data.preprocessing import PreprocessingMapper
from larry.image.config.image_preprocessing_mapper_configs import ImagePreprocessingMapperConfig, \
    URLDownloadingImagePreprocessingMapperConfig, ImageResizingPreprocessingMapperConfig
from larry.image.data import transforms as larry_image_transforms


class ImagePreprocessingMapper[I = None, O = None](PreprocessingMapper[I, O]):
    config: ImagePreprocessingMapperConfig

    def __init__(self, config: ImagePreprocessingMapperConfig) -> None:
        super().__init__(config=config)
        self.config = config


class URLDownloadingImagePreprocessingMapper(ImagePreprocessingMapper[str, torch.Tensor | None]):
    config: URLDownloadingImagePreprocessingMapperConfig

    def __init__(self, config: URLDownloadingImagePreprocessingMapperConfig) -> None:
        super().__init__(config=config)
        self.config = config

    def preprocess_example(self, example: str) -> torch.Tensor | None:
        """
        Download an image from a URL with retry logic.

        Args:
            example: URL to download from
        Returns:
            PIL Image or None if download failed
        """
        response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.get(
                    example,
                    timeout=self.config.timeout,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ImagePreprocessor/1.0)"},
                )
                response.raise_for_status()
                content = response.content
                return TVF.pil_to_tensor(Image.open(io.BytesIO(content)).convert("RGB")).to(torch.uint8)
            except:
                if response is not None:
                    self.log.critical(
                        f"Failure to download image: {response.status_code} - {response.content} for url: {example}")
                else:
                    self.log.critical(f"Failure to download image (can't tell you why though)")
                if attempt == self.config.max_retries:
                    return None
        return None


class ImageResizingPreprocessingMapper(ImagePreprocessingMapper[torch.Tensor, torch.Tensor]):
    config: ImageResizingPreprocessingMapperConfig

    def __init__(self, config: ImageResizingPreprocessingMapperConfig) -> None:
        super().__init__(config=config)
        self.config = config

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
        features = dataset.features.copy()
        for col in self.remove_columns:
            features.pop(col, None)
        features.update({self.config.output_column: datasets.Image()})
        return features

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, Sequence[Any]]:
        images = torch.stack(
            [encode_jpeg(self.preprocess_example(img), quality=95) for img in batch[self.config.input_column]])
        # images = images.permute(0, 2, 3, 1).contiguous().numpy()
        images = images.to(torch.uint8)
        images = images.numpy().tobytes()
        return {self.config.output_column: images}

    def preprocess_example(self, example: torch.Tensor) -> torch.Tensor:
        return self.transform(example)
