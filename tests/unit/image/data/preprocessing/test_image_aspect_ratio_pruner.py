import io

import pytest
from datasets import Dataset, Image
from PIL import Image as PILImage

from larry.common.config.data.preprocessing.preprocessor_configs import ColumnCastingPreprocessorConfig
from larry.common.data.preprocessing.preprocessors import ColumnCastingPreprocessor
from larry.image.config.preprocessing.image_pruner_configs import ImageAspectRatioPrunerConfig
from larry.image.data.preprocessing.image_pruners import ImageAspectRatioPruner
from tests.helpers.common import provenance

SIZES = [(512, 512), (2048, 512), (2100, 512), (512, 2100), (40, 12)]


def _jpeg(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), (90, 120, 150)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {"id": [str(i) for i in range(len(SIZES))], "image": [{"bytes": _jpeg(w, h), "path": None} for w, h in SIZES]}
    ).cast_column("image", Image())


@pytest.mark.parametrize("decode", [True, False])
def test_images_wider_or_taller_than_the_limit_are_pruned(decode: bool) -> None:
    dataset = _dataset()
    if not decode:
        dataset = ColumnCastingPreprocessor(
            provenance(), ColumnCastingPreprocessorConfig(column_casts={"image": "image_no_decode"})
        ).preprocess_dataset(dataset)

    kept = ImageAspectRatioPruner(provenance(), ImageAspectRatioPrunerConfig(num_proc=1)).preprocess_dataset(dataset)

    assert kept["id"] == ["0", "1", "4"]


def test_the_cast_leaves_image_bytes_undecoded() -> None:
    cast = ColumnCastingPreprocessor(
        provenance(), ColumnCastingPreprocessorConfig(column_casts={"image": "image_no_decode"})
    ).preprocess_dataset(_dataset())

    assert isinstance(cast[0]["image"]["bytes"], bytes)
