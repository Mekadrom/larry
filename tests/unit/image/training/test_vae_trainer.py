from pathlib import Path
from typing import Any

import pytest
import torch

from larry.image.config.data.image_data_configs import ImageMixtureConfig, ImageSourceConfig
from larry.image.config.training.image_trainer_configs import ImageVAETrainerConfig, ImageVAEVizConfig
from larry.image.model.vae import ImageVAE
from larry.image.training.vae_trainer import ImageVAETrainer
from tests.helpers.common import provenance, reset_accelerate_state, spawn_launcher, write_parquet_split
from tests.helpers.metrics import RecordingMetrics
from tests.unit.image.data.test_indexed_image_dataset import _png
from tests.unit.image.model.test_vae import TINY


class ImageRecordingMetrics(RecordingMetrics):
    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        self.images: list[str] = []

    def add_image(self, tag: str, *args: Any, **kwargs: Any) -> None:
        self.images.append(tag)


def _sources(root: Path, written: bool = False) -> list[ImageSourceConfig]:
    sizes = [(64, 48), (40, 90), (32, 32), (50, 60)]
    for split in () if written else ("train", "test"):
        write_parquet_split(root, split, {
            "id": [f"{split}{i}" for i in range(len(sizes))],
            "row": list(range(len(sizes))),
            "image": [{"bytes": _png(w, h), "path": f"{i}.png"} for i, (w, h) in enumerate(sizes)],
            "text": ["unused"] * len(sizes),
        })
    return [ImageSourceConfig(name="toy", index_path=str(root), split="train", source_split="train")]


def _trainer(tmp_path: Path, written: bool = False, reset: bool = True, **overrides: Any) -> ImageVAETrainer:
    if reset:
        reset_accelerate_state()
    sources = _sources(tmp_path / "data", written)
    eval_sources = [ImageSourceConfig(name="toy", index_path=sources[0].index_path, split="test", source_split="test")]
    config = ImageVAETrainerConfig(**{
        "mixture_config": ImageMixtureConfig(
            train_sources=sources, eval_sources=eval_sources, crop_size=32, pool_size=8,
        ),
        "max_steps": 3,
        "train_batch_size": 2,
        "eval_batch_size": 3,
        "num_dataloader_workers": 0,
        "mixed_precision": "no",
        "logging_steps": 1,
        "save_steps": 3,
        "eval_steps": 3,
        "viz_steps": 3,
        "lr": 1e-3,
        "viz": ImageVAEVizConfig(examples_per_source=2),
        **overrides,
    })
    torch.manual_seed(0)
    return ImageVAETrainer(provenance(), ImageVAE(TINY), config, ImageRecordingMetrics(str(tmp_path / "run")))


@pytest.mark.parametrize("adversarial_weight", [0.0, 0.1])
def test_a_run_trains_the_vae_and_discriminator_and_logs_every_term(tmp_path: Path, adversarial_weight: float) -> None:
    from larry.image.config.criterion.image_criterion_configs import ImageVAECriterionConfig

    trainer = _trainer(tmp_path, criterion=ImageVAECriterionConfig(
        adversarial_weight=adversarial_weight, equivariance_weight=1.0
    ))
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}

    trainer.run()

    model = trainer.accelerator.unwrap_model(trainer.model)
    changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
    for module in ("encoder", "decoder", "discriminator"):
        assert any(name.startswith(module) for name in changed), module

    scalars = trainer.metrics.scalars
    for term in ("reconstruction", "frequency", "kl", "equivariance", "adversarial", "discriminator"):
        assert f"train/{term}" in scalars and f"eval/{term}" in scalars and f"eval/{term}/toy" in scalars
    assert "eval/psnr" in scalars and "eval/psnr/toy" in scalars and "viz/psnr" in scalars
    assert any(tag.endswith("/reconstruction") for tag in trainer.metrics.images)


def test_the_trainer_keeps_only_image_columns_and_samples_eval_transforms_deterministically(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)

    dataset = trainer.dataset_dict["train"]
    trainer.model.eval()
    first, second = trainer.sample_transform(), trainer.sample_transform()

    assert set(dataset.indexes[0].column_names) == {"id", "row", "image"}
    assert first == second


def _train_on_two_processes(log_dir: str) -> None:
    from accelerate import PartialState

    from larry.image.config.criterion.image_criterion_configs import ImageVAECriterionConfig

    reset_accelerate_state()
    PartialState(cpu=True)
    trainer = _trainer(Path(log_dir), written=True, reset=False, max_steps=2, eval_steps=None, viz_steps=None, save_steps=2,
                       criterion=ImageVAECriterionConfig(adversarial_weight=0.1, equivariance_weight=1.0))
    trainer.run()
    assert trainer.train_state.global_step == 2


def test_the_frozen_discriminator_pass_trains_under_ddp(tmp_path: Path) -> None:
    _sources(tmp_path / "data")

    spawn_launcher(_train_on_two_processes, args=(str(tmp_path),))
