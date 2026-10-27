import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from larry.common.config.config import RunRecord
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.training.trainers import MODEL_CONFIG_FILE, StepOutput
from tests.helpers.common import reset_accelerate_state
from tests.helpers.metrics import NullMetrics, RecordingMetrics
from tests.unit.common.training.toy_training import ToyModel, ToyTrainer, toy_config, toy_trainer, train_toy


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


def test_a_run_records_its_rendered_config_and_cli_in_the_run_dir_and_metrics(tmp_path: Path) -> None:
    metrics = RecordingMetrics(str(tmp_path))
    trainer = ToyTrainer(
        Provenance(config={"trainer": {"type": "ToyTrainerConfig"}}, cli="training -f toy.yaml"),
        ToyModel(),
        toy_config(max_steps=1),
        metrics,
    )

    trainer.run()

    assert yaml.safe_load((tmp_path / RunRecord.CONFIG_FILE).read_text()) == {"trainer": {"type": "ToyTrainerConfig"}}
    assert (tmp_path / RunRecord.CLI_FILE).read_text() == "training -f toy.yaml\n"
    logged = {tag: text for tag, text, _ in metrics.text}
    assert "type: ToyTrainerConfig" in logged["run/config"]
    assert "training -f toy.yaml" in logged["run/cli"]
    assert "type: ToyTrainerConfig" in logged["run/trainer_config"]


def test_a_run_writes_its_model_config_once_and_keeps_an_existing_one(tmp_path: Path) -> None:
    def start() -> ToyTrainer:
        return ToyTrainer(
            Provenance(config={"trainer": {"type": "ToyTrainerConfig"}}, cli="training -f toy.yaml"),
            ToyModel(),
            toy_config(max_steps=1),
            RecordingMetrics(str(tmp_path)),
        )

    trainer = start()

    written = json.loads((tmp_path / MODEL_CONFIG_FILE).read_text())
    assert written == trainer.accelerator.unwrap_model(trainer.model).config.to_dict()
    assert written["type"] == "ModelConfig"
    assert not any((tmp_path / name / MODEL_CONFIG_FILE).exists() for name in ("checkpoint-0", "checkpoint-1"))

    (tmp_path / MODEL_CONFIG_FILE).write_text('{"kept": true}')
    start()

    assert json.loads((tmp_path / MODEL_CONFIG_FILE).read_text()) == {"kept": True}


class ClockedToyTrainer(ToyTrainer):
    clock = [0.0]

    def train_forward_pass(self, batch: Any) -> Any:
        ClockedToyTrainer.clock[0] += 1.0
        return super().train_forward_pass(batch)

    def evaluate(self) -> Any:
        ClockedToyTrainer.clock[0] += 100.0
        return super().evaluate()

    def save(self, step: int) -> None:
        ClockedToyTrainer.clock[0] += 1000.0
        super().save(step)


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    import larry.common.training.trainers as trainers_module

    ClockedToyTrainer.clock = [0.0]
    monkeypatch.setattr(trainers_module.time, "monotonic", lambda: ClockedToyTrainer.clock[0])
    return ClockedToyTrainer.clock


def test_training_time_counts_steps_and_excludes_evaluation_and_saves(tmp_path: Path, fake_clock: list[float]) -> None:
    trainer = train_toy(tmp_path, ClockedToyTrainer, max_steps=6, eval_steps=2, save_steps=3)

    assert trainer.train_state.train_seconds == pytest.approx(6.0)


def test_training_time_resumes_from_the_checkpoint_and_skips_the_pause(tmp_path: Path, fake_clock: list[float]) -> None:
    train_toy(tmp_path / "part", ClockedToyTrainer, max_steps=4, save_steps=4)
    fake_clock[0] += 5000.0
    reset_accelerate_state()

    resumed = train_toy(tmp_path / "rest", ClockedToyTrainer, resume_dir=str(tmp_path / "part" / "checkpoint-4"),
                        max_steps=9, save_steps=4)

    assert resumed.train_state.train_seconds == pytest.approx(9.0)
