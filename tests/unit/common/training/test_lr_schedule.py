import tempfile
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from accelerate import debug_launcher
from accelerate.state import AcceleratorState

from larry.common.metrics.metrics import MetricDict
from tests.unit.common.training.test_step_based_training import NullMetrics, ToyModel, ToyTrainer, ToyTrainerConfig, \
    provenance

WARMUP = 4
MAX_STEPS = 6


class LRRecordingToyTrainer(ToyTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.lrs: list[float] = []
        super().__init__(*args, **kwargs)

    def after_optimizer_step(self, loss: torch.Tensor, step: int) -> MetricDict | None:
        self.lrs.append(float(self.schedulers[0].get_last_lr()[0]))
        return super().after_optimizer_step(loss, step)


def _train_and_check_lr(gradient_accumulation: int) -> None:
    config = ToyTrainerConfig(
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP,
        lr=1.0,
        lr_scheduler_name="constant",
        gradient_accumulation=gradient_accumulation,
        train_batch_size=2,
        eval_batch_size=2,
        num_dataloader_workers=0,
        save_steps=1000,
        logging_steps=1000,
        optimizer_name="adamw",
        mixed_precision="no",
        seed=5,
    )
    with tempfile.TemporaryDirectory() as log_dir:
        trainer = LRRecordingToyTrainer(provenance(), ToyModel(), config, NullMetrics(log_dir))
        trainer.run()

    expected = [min(1.0, 1e-8 + (1.0 - 1e-8) * step / WARMUP) for step in range(1, MAX_STEPS + 1)]
    assert trainer.lrs == pytest.approx(expected), (trainer.accelerator.process_index, trainer.lrs)
    assert trainer.schedulers[0].scheduler.last_epoch == MAX_STEPS


@pytest.fixture(autouse=True)
def fresh_accelerate_state() -> Iterator[None]:
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_warmup_counts_optimizer_steps_in_one_process(gradient_accumulation: int) -> None:
    _train_and_check_lr(gradient_accumulation)


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_warmup_counts_optimizer_steps_not_processes(gradient_accumulation: int) -> None:
    debug_launcher(lambda: _train_and_check_lr(gradient_accumulation), num_processes=2)
