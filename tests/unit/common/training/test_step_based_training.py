from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from larry.common.training.trainers import StepOutput
from tests.helpers.common import reset_accelerate_state
from tests.helpers.metrics import NullMetrics
from tests.unit.common.training.toy_training import ToyTrainer, toy_trainer, train_toy


class EvalRecordingToyTrainer(ToyTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.eval_steps_seen: list[int] = []
        super().__init__(*args, **kwargs)

    def evaluate(self) -> torch.Tensor:
        self.eval_steps_seen.append(self.train_state.global_step)
        return super().evaluate()


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_training_stops_at_max_steps_and_cycles_past_the_end_of_the_data(
        tmp_path: Path, gradient_accumulation: int
) -> None:
    trainer = train_toy(tmp_path, max_steps=9, gradient_accumulation=gradient_accumulation)

    assert trainer.train_state.global_step == 9
    assert len(trainer.model.seen) == 9 * gradient_accumulation
    assert (tmp_path / "checkpoint-9").is_dir()


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_resumed_run_sees_the_same_batches_as_an_uninterrupted_one(
        tmp_path: Path, gradient_accumulation: int
) -> None:
    uninterrupted = train_toy(tmp_path / "full", max_steps=9, gradient_accumulation=gradient_accumulation)

    train_toy(tmp_path / "part", max_steps=4, gradient_accumulation=gradient_accumulation)
    reset_accelerate_state()
    resumed = train_toy(
        tmp_path / "rest",
        resume_dir=str(tmp_path / "part" / "checkpoint-4"),
        max_steps=9,
        gradient_accumulation=gradient_accumulation,
    )

    assert resumed.train_state.global_step == 9
    assert resumed.model.seen == uninterrupted.model.seen[4 * gradient_accumulation:]


def test_checkpoints_are_named_for_completed_steps_and_the_last_is_written_once(tmp_path: Path) -> None:
    train_toy(tmp_path, max_steps=9, save_steps=3)

    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-3", "checkpoint-6", "checkpoint-9"]


def test_a_final_checkpoint_is_written_when_max_steps_is_not_a_save_step(tmp_path: Path) -> None:
    train_toy(tmp_path, max_steps=7, save_steps=3)

    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-3", "checkpoint-6", "checkpoint-7"]


def test_a_fresh_run_evaluates_before_training_and_after_every_eval_steps_completed(tmp_path: Path) -> None:
    trainer = train_toy(tmp_path, EvalRecordingToyTrainer, max_steps=9, eval_steps=4)

    assert trainer.eval_steps_seen == [0, 4, 8]


def test_a_resumed_run_skips_the_baseline_evaluation(tmp_path: Path) -> None:
    train_toy(tmp_path / "part", max_steps=4, save_steps=4)
    reset_accelerate_state()

    resumed = train_toy(
        tmp_path / "rest",
        EvalRecordingToyTrainer,
        resume_dir=str(tmp_path / "part" / "checkpoint-4"),
        max_steps=9,
        eval_steps=4,
    )

    assert resumed.eval_steps_seen == [8]


class ClipCountingToyTrainer(ToyTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.clip_calls = 0
        super().__init__(*args, **kwargs)

    def clip_grads(self) -> Any:
        self.clip_calls += 1
        return super().clip_grads()


@pytest.mark.parametrize("gradient_accumulation", [1, 3])
def test_grads_are_clipped_once_per_optimizer_step(tmp_path: Path, gradient_accumulation: int) -> None:
    trainer = train_toy(tmp_path, ClipCountingToyTrainer, max_steps=4, gradient_accumulation=gradient_accumulation)

    assert trainer.clip_calls == 4


class LossRecordingToyTrainer(ToyTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.microbatch_losses: list[float] = []
        self.step_losses: list[float] = []
        super().__init__(*args, **kwargs)

    def train_forward_pass(self, batch: Any) -> StepOutput:
        output = super().train_forward_pass(batch)
        self.microbatch_losses.append(output.loss.item())
        return output

    def after_optimizer_step(self, loss: torch.Tensor, step: int) -> Any:
        self.step_losses.append(loss.item())
        return super().after_optimizer_step(loss, step)


def test_each_optimizer_step_reports_the_mean_loss_of_its_microbatches(tmp_path: Path) -> None:
    trainer = train_toy(tmp_path, LossRecordingToyTrainer, max_steps=3, gradient_accumulation=2)

    expected = [sum(trainer.microbatch_losses[i:i + 2]) / 2 for i in range(0, 6, 2)]
    assert trainer.step_losses == pytest.approx(expected)


class CountingIterable:
    def __init__(self, inner: Iterable[Any]) -> None:
        self.inner = inner
        self.taken = 0

    def __iter__(self) -> Iterator[Any]:
        for item in self.inner:
            self.taken += 1
            yield item


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_training_takes_no_batch_past_the_last_step(tmp_path: Path, gradient_accumulation: int) -> None:
    trainer = toy_trainer(
        ToyTrainer, NullMetrics(str(tmp_path)), max_steps=5, gradient_accumulation=gradient_accumulation
    )
    loader = CountingIterable(trainer.train_loader)
    trainer.train_loader = loader

    trainer.run()

    assert loader.taken == 5 * gradient_accumulation


def test_adamw_uses_the_configured_betas(tmp_path: Path) -> None:
    trainer = toy_trainer(ToyTrainer, NullMetrics(str(tmp_path)), max_steps=1, adamw_betas=(0.8, 0.95))

    assert [group["betas"] for o in trainer.optimizers for group in o.param_groups] == [(0.8, 0.95)]


def test_optimizer_config_reaches_the_configured_optimizer(tmp_path: Path) -> None:
    adamw = toy_trainer(ToyTrainer, NullMetrics(str(tmp_path / "adamw")), max_steps=1, optimizer_config={"eps": 1e-5})
    reset_accelerate_state()
    muon = toy_trainer(
        ToyTrainer,
        NullMetrics(str(tmp_path / "muon")),
        max_steps=1,
        optimizer_name="muon",
        optimizer_config={"momentum": 0.9},
    )

    assert [group["eps"] for group in adamw.optimizers[0].param_groups] == [1e-5]
    assert [group["momentum"] for group in muon.optimizers[0].param_groups] == [0.9]
    assert [group["betas"] for group in muon.optimizers[1].param_groups] == [(0.9, 0.999)]


class FrozenBiasToyTrainer(ToyTrainer):
    def make_optimizers(self, model: torch.nn.Module) -> Any:
        model.get_submodule("linear").bias.requires_grad_(False)
        return super().make_optimizers(model)


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_frozen_params_are_left_out_of_the_optimizers(tmp_path: Path, optimizer_name: str) -> None:
    trainer = toy_trainer(FrozenBiasToyTrainer, NullMetrics(str(tmp_path)), max_steps=1, optimizer_name=optimizer_name)

    optimized = [p for o in trainer.optimizers for group in o.param_groups for p in group["params"]]
    assert len(optimized) == 1
    assert all(p.requires_grad for p in optimized)
