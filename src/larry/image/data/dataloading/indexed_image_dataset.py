from typing import Any

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torchcodec.decoders import ImageReadMode, decode_image
from torchvision.transforms.v2 import functional as TVF

from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.data.dataloading.worker_random import WorkerRandom
from larry.image.config.data.image_data_configs import ImageSourceConfig


class IndexedImageDataset(IndexedDataset[ImageSourceConfig]):
    def __init__(
            self,
            configs: list[ImageSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            crop_size: int,
            train: bool = True,
            random_flip: bool = True,
            seed: int = 0,
            process_index: int = 0,
    ) -> None:
        super().__init__(configs, indexes, sources)
        self.crop_size = crop_size
        self.train = train
        self.random_flip = random_flip
        self.random = WorkerRandom(seed, process_index)

    def get_data(self, source_idx: int, entry: dict[Any, Any]) -> Any:
        return entry[self.configs[source_idx].image_column]

    @staticmethod
    def decode(data: Any) -> torch.Tensor:
        # undecoded image features arrive as {"bytes", "path"}; decoded ones as PIL images
        if isinstance(data, Image.Image):
            return TVF.pil_to_tensor(data.convert("RGB"))
        return decode_image(torch.frombuffer(bytearray(data["bytes"]), dtype=torch.uint8), mode=ImageReadMode.RGB)

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        image = TVF.resize(image, [self.crop_size], antialias=True)
        height, width = image.shape[-2:]
        if not self.train:
            return TVF.center_crop(image, [self.crop_size, self.crop_size])

        rng = self.random.get()
        top = rng.randint(0, height - self.crop_size)
        left = rng.randint(0, width - self.crop_size)
        image = TVF.crop(image, top, left, self.crop_size, self.crop_size)
        if self.random_flip and rng.random() < 0.5:
            image = TVF.horizontal_flip(image)
        return image

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        # [-1, 1], the usual VAE input range
        image = self.crop(self.decode(data)).float() / 127.5 - 1
        return {"image": image, "source": source_idx}


class ImageCollator:
    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "images": torch.stack([item["image"] for item in items]),
            "source_ids": torch.tensor([item["source"] for item in items], dtype=torch.long),
        }
