from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate.state import AcceleratorState

from larry.common.training.trainers import StepOutput

from tests.unit.common.training.test_step_based_training import NullMetrics, ToyModel, ToyTrainer, ToyTrainerConfig


class RecordingMetrics(NullMetrics):
    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag: str, value: float, step: int, **kwargs: Any) -> None:
        self.scalars[tag] = value


class GroupedToyTrainer(ToyTrainer):
    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self.model(**batch)
        sample_losses = output.logits.squeeze(-1).pow(2)
        return StepOutput(loss=sample_losses.mean(), metrics=output.metrics, sample_losses=sample_losses)

    def eval_group_names(self) -> list[str]:
        return ["even", "odd"]

    def eval_group_totals(
            self, batch: Mapping[str, Any], output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        assert output.sample_losses is not None
        return batch["idx"] % 2, output.sample_losses, 1 + batch["idx"] % 3


@pytest.fixture(autouse=True)
def fresh_accelerate_state() -> Iterator[None]:
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def _trainer(trainer_type: type[ToyTrainer], tmp_path: Path) -> tuple[ToyTrainer, RecordingMetrics]:
    config = ToyTrainerConfig(
        max_steps=1,
        train_batch_size=2,
        eval_batch_size=2,
        num_dataloader_workers=0,
        optimizer_name="adamw",
        mixed_precision="bf16",
    )
    metrics = RecordingMetrics(str(tmp_path))
    trainer = trainer_type(ToyModel(), config, metrics)
    with torch.no_grad():
        trainer.model.linear.weight.fill_(1.0)
        trainer.model.linear.bias.zero_()
    return trainer, metrics


def test_eval_logs_a_token_weighted_loss_per_group_and_overall(tmp_path: Path) -> None:
    trainer, metrics = _trainer(GroupedToyTrainer, tmp_path)

    overall = trainer.evaluate()

    assert metrics.scalars["eval/loss/even"] == pytest.approx((0 + 4) / (1 + 3), rel=1e-2)
    assert metrics.scalars["eval/loss/odd"] == pytest.approx((1 + 9) / (2 + 1), rel=1e-2)
    assert metrics.scalars["eval/loss"] == pytest.approx((0 + 1 + 4 + 9) / (1 + 2 + 3 + 1), rel=1e-2)
    assert float(overall) == pytest.approx(metrics.scalars["eval/loss"])


def test_eval_without_groups_logs_only_the_overall_loss(tmp_path: Path) -> None:
    trainer, metrics = _trainer(ToyTrainer, tmp_path)

    trainer.evaluate()

    assert "eval/loss" in metrics.scalars
    assert not [tag for tag in metrics.scalars if tag.startswith("eval/loss/")]
