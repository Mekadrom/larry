import math
import tempfile
from collections.abc import Iterator
from pathlib import Path
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


def _run(log_dir: str, max_steps: int, resume_dir: str | None = None, **schedule: Any) -> LRRecordingToyTrainer:
    config = ToyTrainerConfig(
        max_steps=max_steps,
        lr=1.0,
        train_batch_size=2,
        eval_batch_size=2,
        num_dataloader_workers=0,
        save_steps=1000,
        logging_steps=1000,
        optimizer_name="adamw",
        mixed_precision="no",
        seed=5,
        **schedule,
    )
    trainer = LRRecordingToyTrainer(provenance(), ToyModel(), config, NullMetrics(log_dir), resume_dir=resume_dir)
    trainer.run()
    return trainer


def _cosine(step: int, start: int, end: int, min_lr: float) -> float:
    return min_lr + (1.0 - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (step - start) / (end - start)))


def test_cosine_decays_from_the_end_of_warmup_by_default(tmp_path: Path) -> None:
    trainer = _run(str(tmp_path), 10, warmup_steps=2, lr_scheduler_name="cosine", min_lr=0.1)

    expected = [0.5] + [_cosine(step, 2, 10, 0.1) for step in range(2, 11)]
    assert trainer.lrs == pytest.approx(expected)


def test_a_constant_run_resumes_into_a_cosine_decay_that_starts_at_the_resume_step(tmp_path: Path) -> None:
    _run(str(tmp_path / "stable"), 6, warmup_steps=2, lr_scheduler_name="constant")
    AcceleratorState._reset_state(reset_partial_state=True)

    decay = _run(
        str(tmp_path / "decay"), 10, resume_dir=str(tmp_path / "stable" / "checkpoint-6"),
        warmup_steps=2, lr_scheduler_name="cosine", min_lr=0.1, decay_start_step=6,
    )

    assert decay.lrs == pytest.approx([_cosine(step, 6, 10, 0.1) for step in range(7, 11)])
    assert decay.lrs[-1] == pytest.approx(0.1)
