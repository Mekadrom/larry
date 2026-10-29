import io
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image

from larry.image.config.model.image_model_configs import ImageVAEConfig, PatchDiscriminatorConfig
from larry.image.config.preprocessing.image_mapper_configs import LarryImageVAELatentExtractingMapperConfig
from larry.image.data.preprocessing import image_mappers
from larry.image.data.preprocessing.image_mappers import LarryImageVAELatentExtractingMapper
from larry.image.model.vae import ImageVAE
from tests.helpers.common import provenance

TINY = ImageVAEConfig(
    base_channels=16, channel_mults=[1, 2, 2], n_res_blocks=1, latent_channels=4, norm_groups=4,
    discriminator=PatchDiscriminatorConfig(base_channels=8, n_layers=2, norm_groups=4),
)


def _png(width: int, height: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    torch.manual_seed(0)
    ImageVAE(TINY).save_pretrained(tmp_path)
    monkeypatch.setitem(image_mappers._VAES, str(tmp_path), ImageVAE.from_pretrained(str(tmp_path)).eval())
    return tmp_path


def _mapper(checkpoint: Path, **overrides: object) -> LarryImageVAELatentExtractingMapper:
    config = LarryImageVAELatentExtractingMapperConfig(
        **{"vae_checkpoint_path": str(checkpoint), "cache_results": False, **overrides}
    )
    return LarryImageVAELatentExtractingMapper(provenance(), config)


def test_latents_keep_the_aspect_ratio_on_a_multiple_of_the_size_step(checkpoint: Path) -> None:
    mapper = _mapper(checkpoint, image_size=32, size_multiple=8)

    out = mapper.preprocess_batch({"image": [_png(90, 60, 0), _png(40, 40, 1)]})

    assert out["latent_shape"] == [[4, 8, 12], [4, 8, 8]]
    assert [len(x) for x in out["latents"]] == [4 * 8 * 12, 4 * 8 * 8]
    assert out["latents"][0].dtype == np.float16


def test_latents_are_the_vae_mean_of_the_prepared_image(checkpoint: Path) -> None:
    mapper = _mapper(checkpoint, image_size=32, size_multiple=8)
    image = _png(48, 32, 2)

    out = mapper.preprocess_batch({"image": [image]})
    with torch.inference_mode():
        mean, _ = ImageVAE.from_pretrained(str(checkpoint)).eval().encode(mapper.prepare(image)[None])

    expected = mean[0].numpy().astype(np.float16).reshape(-1)
    assert np.array_equal(out["latents"][0], expected)


def test_the_vae_checkpoint_is_recorded_in_provenance(checkpoint: Path) -> None:
    mapper = _mapper(checkpoint)

    mapper.mutate_provenance()

    assert str(checkpoint) in mapper.provenance.ancestry.models


def test_the_mapper_adds_latent_columns_to_a_dataset(checkpoint: Path) -> None:
    dataset = Dataset.from_dict({"image": [_png(64, 32, 3)], "caption": ["noise"]})

    mapped = _mapper(checkpoint, image_size=16, size_multiple=8, num_proc=None).preprocess_dataset(dataset)

    assert mapped["latent_shape"] == [[4, 4, 8]]
    assert mapped["caption"] == ["noise"]
