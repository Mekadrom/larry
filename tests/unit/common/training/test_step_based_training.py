import dataclasses
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate.state import AcceleratorState
from torch import nn

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.criteria.criterion import Criterion
from larry.common.metrics.metrics import Metrics
from larry.common.model.model import LarryModelOutput
from larry.common.training.trainers import LarryTrainer
from larry.common.training.training_model import LarryModel
from larry.common.training.viz_callbacks import VizCallback


@dataclasses.dataclass(kw_only=True)
class ToyTrainerConfig(LarryTrainerConfig):
    ...


@dataclasses.dataclass(kw_only=True)
class ToyCriterionConfig(CriterionConfig):
    ...


class ToyModel(LarryModel[ModelConfig]):
    def __init__(self) -> None:
        super().__init__(ModelConfig(d_model=1, max_position_embeddings=1))
        self.linear = nn.Linear(1, 1)
        self.seen: list[list[int]] = []

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> LarryModelOutput:
        if self.training:
            self.seen.append(idx.tolist())
        return LarryModelOutput(logits=self.linear(x.float().unsqueeze(-1)))


class ToyCriterion(Criterion[ToyCriterionConfig, ToyModel]):
    def forward(self, model_output: LarryModelOutput, **batch_kwargs: Any) -> torch.Tensor:
        return model_output.logits.pow(2).mean()


class NullMetrics(Metrics):
    def add_scalar(self, *args: Any, **kwargs: Any) -> None: ...
    def add_image(self, *args: Any, **kwargs: Any) -> None: ...
    def add_audio(self, *args: Any, **kwargs: Any) -> None: ...
    def add_figure(self, *args: Any, **kwargs: Any) -> None: ...
    def add_grad_norms(self, *args: Any, **kwargs: Any) -> None: ...
    def add_histogram(self, *args: Any, **kwargs: Any) -> None: ...
    def add_hparams(self, *args: Any, **kwargs: Any) -> None: ...
    def add_text(self, *args: Any, **kwargs: Any) -> None: ...
    def close(self) -> None: ...
    def flush(self) -> None: ...


class ToyTrainer(LarryTrainer[ToyModel, ToyTrainerConfig]):
    def make_dataset_dict(self) -> Mapping[str, Any]:
        rows = [{"x": float(i), "idx": i} for i in range(11)]
        return {"train": rows, "validation": rows[:4]}

    def make_criterion(self) -> Criterion:
        return ToyCriterion(ToyCriterionConfig(), self.model)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def fresh_accelerate_state() -> Iterator[None]:
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def _train(log_dir: Path, max_steps: int, gradient_accumulation: int, resume_dir: str | None = None) -> ToyTrainer:
    config = ToyTrainerConfig(
        max_steps=max_steps,
        gradient_accumulation=gradient_accumulation,
        train_batch_size=2,
        eval_batch_size=2,
        num_dataloader_workers=0,
        save_steps=1000,
        logging_steps=1000,
        optimizer_name="adamw",
        mixed_precision="bf16",
        seed=5,
    )
    trainer = ToyTrainer(ToyModel(), config, NullMetrics(str(log_dir)), resume_dir=resume_dir)
    trainer.run()
    return trainer


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_training_stops_at_max_steps_and_cycles_past_the_end_of_the_data(
        tmp_path: Path, gradient_accumulation: int
) -> None:
    trainer = _train(tmp_path, max_steps=9, gradient_accumulation=gradient_accumulation)

    assert trainer.train_state.global_step == 9
    assert len(trainer.model.seen) == 9 * gradient_accumulation
    assert (tmp_path / "checkpoint-9").is_dir()


@pytest.mark.parametrize("gradient_accumulation", [1, 2])
def test_resumed_run_sees_the_same_batches_as_an_uninterrupted_one(
        tmp_path: Path, gradient_accumulation: int
) -> None:
    uninterrupted = _train(tmp_path / "full", max_steps=9, gradient_accumulation=gradient_accumulation)

    _train(tmp_path / "part", max_steps=4, gradient_accumulation=gradient_accumulation)
    AcceleratorState._reset_state(reset_partial_state=True)
    resumed = _train(
        tmp_path / "rest",
        max_steps=9,
        gradient_accumulation=gradient_accumulation,
        resume_dir=str(tmp_path / "part" / "checkpoint-4"),
    )

    assert resumed.train_state.global_step == 9
    assert resumed.model.seen == uninterrupted.model.seen[4 * gradient_accumulation:]
