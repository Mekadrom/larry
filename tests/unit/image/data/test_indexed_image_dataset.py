import io
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from larry.common.data.dataloading.indexed_dataset import load_index
from larry.image.config.data.image_data_configs import ImageSourceConfig
from larry.image.data.dataloading.indexed_image_dataset import ImageCollator, IndexedImageDataset
from tests.helpers.common import write_parquet_split

SIZES = [(64, 48), (40, 90), (32, 32)]


def _png(width: int, height: int) -> bytes:
    gradient = torch.arange(width).repeat(height, 1) * 255 // max(width - 1, 1)
    pixels = torch.stack([gradient, gradient.flip(1), torch.full_like(gradient, 128)], dim=-1).to(torch.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels.numpy()).save(buffer, format="PNG")
    return buffer.getvalue()


def _dataset(tmp_path: Path, **kwargs: Any) -> IndexedImageDataset:
    root = write_parquet_split(tmp_path, "train", {
        "id": [str(i) for i in range(len(SIZES))],
        "image": [{"bytes": _png(w, h), "path": f"{i}.png"} for i, (w, h) in enumerate(SIZES)],
    })
    config = ImageSourceConfig(name="toy", index_path=str(root))
    return IndexedImageDataset([config], [load_index(config)], [None], **{"crop_size": 32, **kwargs})


def test_every_image_becomes_a_square_crop_in_the_vae_range(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    images = [dataset[i]["image"] for i in range(len(dataset))]

    assert all(image.shape == (3, 32, 32) and image.dtype == torch.float32 for image in images)
    assert all(image.min() >= -1 and image.max() <= 1 for image in images)
    assert images[2][2].allclose(torch.full((32, 32), 128 / 127.5 - 1), atol=1e-6)


def test_eval_crops_are_centered_and_repeatable(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, train=False)

    first, again = dataset[0]["image"], dataset[0]["image"]

    assert torch.equal(first, again)
    # 64x48 resizes to 43x32; the centered crop leaves the red ramp symmetric around the middle
    assert first[0, :, 16].mean() == pytest.approx(0.0, abs=0.1)


def test_training_crops_vary_but_are_seeded(tmp_path: Path) -> None:
    def crops(seed: int) -> list[torch.Tensor]:
        dataset = _dataset(tmp_path, seed=seed)
        return [dataset[1]["image"] for _ in range(8)]

    assert all(torch.equal(a, b) for a, b in zip(crops(3), crops(3), strict=True))
    assert len({tuple(crop[0, :, 0].tolist()) for crop in crops(3)}) > 1


def test_flipping_can_be_turned_off(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, random_flip=False)

    crops = [dataset[0]["image"] for _ in range(8)]

    # without flips the red ramp always increases left to right
    assert all(crop[0, 16, -1] > crop[0, 16, 0] for crop in crops)


def test_decoded_pil_images_are_accepted(tmp_path: Path) -> None:
    image = Image.open(io.BytesIO(_png(48, 64)))

    assert IndexedImageDataset.decode(image).shape == (3, 64, 48)


def test_the_collator_stacks_crops_and_sources(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    batch = ImageCollator()([dataset[i] for i in range(3)])

    assert batch["images"].shape == (3, 3, 32, 32)
    assert batch["source_ids"].tolist() == [0, 0, 0]
