from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from larry.common.criteria.criterion import LossTerm, TermCriterion
from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.model.model import LarryModelOutput
from larry.common.training.trainers import StepOutput
from tests.helpers.metrics import RecordingMetrics
from tests.unit.common.training.toy_training import ToyCriterionConfig, ToyTrainer, toy_trainer


class TwoTermCriterion(TermCriterion[LarryModelOutput, ToyCriterionConfig]):
    def loss_terms(self, model_output: LarryModelOutput, idx: torch.Tensor, **batch: Any) -> dict[str, LossTerm]:
        x = model_output.logits.squeeze(-1).float()
        # "a" is averaged over 1 + idx % 3 tokens per row; "b" counts once per row, and only for odd rows
        odd = (idx % 2).float()
        return {
            "a": LossTerm(weight=1.0, sums=x.square(), counts=(1 + idx % 3).float()),
            "b": LossTerm(weight=0.5, sums=x * odd, counts=odd),
        }


class GroupedToyTrainer(ToyTrainer):
    def make_criterion(self) -> TwoTermCriterion:
        return TwoTermCriterion(ToyCriterionConfig())

    def eval_groupings(self) -> list[tuple[str, list[str]]]:
        return [("", ["even", "odd"]), ("half/", ["low", "high"])]

    def eval_group_ids(self, batch: Mapping[str, Any]) -> list[torch.Tensor] | None:
        return [batch["idx"] % 2, (batch["idx"] >= 2).long()]


class StatisticsToyTrainer(ToyTrainer):
    def eval_statistics(self) -> dict[str, torch.Tensor]:
        return {"rows": torch.zeros(2, device=self.accelerator.device)}

    def update_eval_statistics(
            self, statistics: dict[str, torch.Tensor], batch: Mapping[str, Any], output: StepOutput
    ) -> None:
        assert output.model_output is not None
        statistics["rows"] += torch.stack([(batch["idx"] % 2 == 0).sum(), (batch["idx"] % 2 == 1).sum()]).float()

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        even, odd = statistics["rows"].tolist()
        metrics = MetricDict()
        metrics.add_metric(ScalarMetricEntry(tag="eval/even_rows", scalar=even))
        metrics.add_metric(ScalarMetricEntry(tag="eval/odd_rows", scalar=odd))
        return metrics


def _trainer(trainer_type: type[ToyTrainer], tmp_path: Path) -> tuple[ToyTrainer, RecordingMetrics]:
    metrics = RecordingMetrics(str(tmp_path))
    trainer = toy_trainer(trainer_type, metrics, max_steps=1)
    with torch.no_grad():
        trainer.model.linear.weight.fill_(1.0)
        trainer.model.linear.bias.zero_()
    return trainer, metrics


def test_eval_logs_the_weighted_total_and_every_term_per_group_and_overall(tmp_path: Path) -> None:
    trainer, metrics = _trainer(GroupedToyTrainer, tmp_path)

    overall = trainer.evaluate()

    # eval rows are x = idx = 0..3; "a" sums x^2 over 1 + idx % 3 tokens, "b" sums x over odd rows only
    scalars = metrics.scalars
    assert scalars["eval/a/even"] == pytest.approx((0 + 4) / (1 + 3), rel=1e-2)
    assert scalars["eval/a/odd"] == pytest.approx((1 + 9) / (2 + 1), rel=1e-2)
    assert scalars["eval/b/odd"] == pytest.approx((1 + 3) / 2, rel=1e-2)
    assert "eval/b/even" not in scalars
    assert scalars["eval/loss/even"] == pytest.approx(scalars["eval/a/even"], rel=1e-2)
    assert scalars["eval/loss/odd"] == pytest.approx(scalars["eval/a/odd"] + 0.5 * scalars["eval/b/odd"], rel=1e-2)
    assert scalars["eval/a/half/high"] == pytest.approx((4 + 9) / (3 + 1), rel=1e-2)
    assert scalars["eval/b/half/low"] == pytest.approx(1.0, rel=1e-2)
    assert scalars["eval/a"] == pytest.approx((0 + 1 + 4 + 9) / (1 + 2 + 3 + 1), rel=1e-2)
    assert scalars["eval/loss"] == pytest.approx(scalars["eval/a"] + 0.5 * scalars["eval/b"], rel=1e-2)
    assert float(overall) == pytest.approx(scalars["eval/loss"])


def test_training_logs_every_term_beside_the_total(tmp_path: Path) -> None:
    trainer, metrics = _trainer(GroupedToyTrainer, tmp_path)

    trainer.run()

    assert {"train/loss", "train/a", "train/b"} <= set(metrics.scalars)


def test_a_criterion_without_per_row_terms_logs_only_the_overall_loss(tmp_path: Path) -> None:
    trainer, metrics = _trainer(ToyTrainer, tmp_path)

    trainer.evaluate()

    assert "eval/loss" in metrics.scalars
    assert not [tag for tag in metrics.scalars if tag.startswith("eval/loss/")]


def test_eval_statistics_accumulate_over_every_batch_and_are_logged(tmp_path: Path) -> None:
    trainer, metrics = _trainer(StatisticsToyTrainer, tmp_path)

    trainer.evaluate()

    assert metrics.scalars["eval/even_rows"] == 2
    assert metrics.scalars["eval/odd_rows"] == 2
