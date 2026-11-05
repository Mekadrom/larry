from types import SimpleNamespace
from typing import Any

import torch

from larry.common.training.trainers import LarryTrainer
from larry.voice.config.training.voice_trainer_configs import VocoderTrainerConfig
from tests.unit.common.training.toy_training import toy_config


def _loader(**overrides: Any) -> torch.utils.data.DataLoader:
    config = toy_config(max_steps=1, **overrides)
    stub = SimpleNamespace(
        config=SimpleNamespace(**{**vars(config), "mixture_config": SimpleNamespace(train_sources=[])}),
        make_train_sampler=lambda dataset, weights: [[0, 1]],
        make_collate_fn=lambda: None,
    )
    return LarryTrainer.make_train_dataloader(stub, {"train": list(range(4))})  # type: ignore[arg-type]


def test_workers_prefetch_the_configured_number_of_batches() -> None:
    assert _loader(num_dataloader_workers=2, train_prefetch_factor=6).prefetch_factor == 6


def test_the_prefetch_factor_is_left_unset_without_workers() -> None:
    assert _loader(num_dataloader_workers=0, train_prefetch_factor=6).prefetch_factor is None


def test_trainers_prefetch_two_batches_per_worker_by_default() -> None:
    assert toy_config(max_steps=1).train_prefetch_factor == 2
    assert VocoderTrainerConfig.__dataclass_fields__["train_prefetch_factor"].default == 2
