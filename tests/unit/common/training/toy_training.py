import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Sampler

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.criteria.criterion import Criterion
from larry.common.data.dataloading.samplers import ShuffledBatchSampler
from larry.common.metrics.metrics import Metrics
from larry.common.model.model import LarryModelOutput
from larry.common.training.trainers import LarryTrainer
from larry.common.training.training_model import LarryModel
from larry.common.training.viz_callbacks import VizCallback
from tests.helpers.common import provenance
from tests.helpers.metrics import NullMetrics


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


class ToyCriterion(Criterion[LarryModelOutput, ToyCriterionConfig]):
    def forward(self, model_output: LarryModelOutput, **batch_kwargs: Any) -> torch.Tensor:
        return model_output.logits.pow(2).mean()


class ToyTrainer(LarryTrainer[ToyModel, ToyTrainerConfig]):
    def make_dataset_dict(self) -> Mapping[str, Any]:
        rows = [{"x": float(i), "idx": i} for i in range(11)]
        return {"train": rows, "validation": rows[:4]}

    def load_indexed(self, configs: list[Any], indexes: list[Any], sources: list[Any], is_eval: bool = False) -> Any:
        raise NotImplementedError

    def make_train_sampler(self, dataset: Any, weights: list[float]) -> Sampler:
        return ShuffledBatchSampler(len(dataset), self.config.train_batch_size, self.config.seed)

    def make_eval_sampler(self, dataset: Any) -> Iterable[list[int]]:
        size = self.config.eval_batch_size
        return [list(range(start, min(start + size, len(dataset)))) for start in range(0, len(dataset), size)]

    def make_criterion(self) -> Criterion:
        return ToyCriterion(ToyCriterionConfig())

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        raise NotImplementedError

    def update_provenance(self, provenance: Provenance) -> None:
        ...


def toy_config(**overrides: Any) -> ToyTrainerConfig:
    return ToyTrainerConfig(**{
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "num_dataloader_workers": 0,
        "logging_steps": 1000,
        "optimizer_name": "adamw",
        "seed": 5,
        **overrides,
    })


def toy_trainer[T: ToyTrainer](
        trainer_type: type[T], metrics: Metrics, resume_dir: str | None = None, **overrides: Any
) -> T:
    return trainer_type(provenance(), ToyModel(), toy_config(**overrides), metrics, resume_dir=resume_dir)


def train_toy[T: ToyTrainer](
        log_dir: Path | str, trainer_type: type[T] = ToyTrainer, resume_dir: str | None = None, **overrides: Any
) -> T:
    trainer = toy_trainer(trainer_type, NullMetrics(str(log_dir)), resume_dir, **overrides)
    trainer.run()
    return trainer
