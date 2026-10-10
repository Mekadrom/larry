import dataclasses
import functools
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Self, cast, Concatenate

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import DataLoaderShard
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from datasets import load_dataset
from torch import nn
from torch.optim import Optimizer, Muon, AdamW
from torch.optim.lr_scheduler import LRScheduler, ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.criteria import criterion
from larry.common.metrics.metrics import MetricDict, Metrics, ScalarMetricEntry
from larry.common.optim.qk_clip import QKClipHandler, QKClipCallback
from larry.common.training.callbacks import TrainingCallback, ModelInitLoggingCallback
from larry.common.training.training_model import LarryModel
from larry.common.training.viz_callbacks import VizCallback
from larry.common.utils import git_utils
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class StepOutput:
    loss: torch.Tensor
    metrics: MetricDict


class TrainerBase[M: LarryModel, C: LarryTrainerConfig = LarryTrainerConfig](Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[TrainerBase]]

    _model: M
    _config: C

    train_state: TrainState

    training_callbacks: list[TrainingCallback]
    train_loader: DataLoaderShard
    eval_loader: DataLoaderShard
    optimizers: list[AcceleratedOptimizer]
    schedulers: list[AcceleratedScheduler]

    total_steps: int

    def __init__(self, model: M, config: C, metrics: Metrics, resume_dir: str | None = None) -> None:
        self.log = logging.getLogger(type(self).__name__)

        self._config = config
        self.model_config = model.config

        self.training_callbacks = []

        dataset_dict = self.make_dataset_dict()
        train_loader = self.make_train_dataloader(dataset_dict)
        eval_loader = self.make_eval_dataloader(dataset_dict)

        self.total_steps = math.ceil(len(train_loader) / self.config.gradient_accumulation) * self.config.num_epochs

        optimizers = self.make_optimizers(model)  # e.g. [muon, adamw]
        schedulers = [self.make_lr_scheduler(o) for o in optimizers]

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        )
        self._model, self.train_loader, self.eval_loader, *rest = self.accelerator.prepare(
            model, train_loader, eval_loader, *optimizers, *schedulers
        )

        self.log.info(f"optimizers={optimizers}")
        self.log.info(f"schedulers={schedulers}")

        n_opt = len(optimizers)

        self.optimizers = cast(list[AcceleratedOptimizer], rest[:n_opt])
        self.schedulers = cast(list[AcceleratedScheduler], rest[n_opt:])

        self.log.info(f"self.optimizers={self.optimizers}")
        self.log.info(f"self.schedulers={self.schedulers}")

        self.train_state = TrainState()
        self.load(resume_dir)

        self.metrics = metrics

        self.add_default_training_callbacks()

        self.criterion = self.make_criterion()

        self._step_open = False

    @property
    def model(self) -> M:
        return self._model

    @property
    def config(self) -> C:
        return self._config

    @functools.cached_property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @functools.cached_property
    def git_commit_hash(self) -> str:
        repo_state = git_utils.git_info()
        if repo_state is not None:
            return repo_state.git_commit
        return "head"

    def make_optimizers(self, model: nn.Module) -> list[Optimizer]:
        if self.config.optimizer_name == "muon":
            excluded = set(self.config.muon_first_layer_names + self.config.muon_last_layer_names)
            missing = excluded - {n for n, _ in model.named_parameters()}
            if missing:
                raise ValueError(f"excluded Muon params not found in model: {missing}")

            muon_params = []
            for n, p in model.named_parameters():
                if p.ndim == 2 and n not in excluded:
                    muon_params.append(p)

            other_params = []
            for n, p in model.named_parameters():
                if not (p.ndim == 2 and n not in excluded):
                    other_params.append(p)

            return [
                Muon(muon_params, adjust_lr_fn="match_rms_adamw", **self._optimizer_kwargs()),
                AdamW(other_params, **self._optimizer_kwargs())
            ]
        elif self.config.optimizer_name == "adamw":
            return [AdamW(model.parameters(), **self._optimizer_kwargs())]
        raise ValueError(f"Unrecognized optimizer: {self.config.optimizer_name}")

    def _optimizer_kwargs(self) -> dict[str, Any]:
        return dict(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

    def make_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        if self.config.lr_scheduler_name == "cosine":
            main = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps - self.config.warmup_steps,
                eta_min=self.config.min_lr
            )
        else:
            if self.config.lr_scheduler_name != "constant":
                self.log.warning(
                    f"Unrecognized lr_scheduler: {self.config.lr_scheduler_name}; "
                    f"defaulting to 'constant'"
                )
            main = ConstantLR(optimizer, factor=1.0)

        if self.config.warmup_steps:
            warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.config.warmup_steps)
            return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[self.config.warmup_steps])

        return main

    def load(self, resume_dir: str | None = None) -> None:
        self.accelerator.register_for_checkpointing(self.train_state)
        if resume_dir:
            self.accelerator.load_state(resume_dir)

    def add_default_training_callbacks(self) -> None:
        if self.config.optimizer_name == "muon":
            self.training_callbacks.append(self._make_qk_clip_callback(self.model))
        if self.config.viz_steps:
            self.training_callbacks.append(self.make_viz_callback(self.config.viz_steps))
        self.training_callbacks.append(ModelInitLoggingCallback())

    def _make_qk_clip_callback(self, model: nn.Module) -> QKClipCallback:
        return QKClipCallback(QKClipHandler(
            model,
            tau=self.config.muon_qk_clip_tau,
            probe_every=self.config.muon_qk_clip_probe_every,
            alpha=self.config.muon_qk_clip_alpha
        ))

    def run(self) -> None:
        steps_per_epoch = math.ceil(len(self.train_loader) / self.accelerator.gradient_accumulation_steps)
        with tqdm(initial=self.train_state.global_step,
                  total=steps_per_epoch * self.config.num_epochs,
                  disable=not self.accelerator.is_local_main_process) as pbar:
            for epoch in range(self.train_state.epoch, self.config.num_epochs):
                self.train_state.epoch = epoch
                self.train_loader.set_epoch(epoch)

                self.on_epoch_begin(epoch)

                self.train(pbar=pbar, skip_steps=self.train_state.step_in_epoch)
                self.train_state.step_in_epoch = 0

                eval_loss = None
                if self.config.eval_steps is None:
                    eval_loss = self.evaluate()

                self.on_epoch_end(eval_loss, epoch)

        self.save(self.train_state.global_step)

    def train(self, pbar: tqdm, skip_steps: int) -> None:
        self.model.train()

        skip = skip_steps * self.accelerator.gradient_accumulation_steps
        # noinspection PyTypeChecker
        loader = self.accelerator.skip_first_batches(self.train_loader, skip) if skip else self.train_loader

        for batch in loader:
            metrics_step = self.train_state.global_step

            step_metrics = MetricDict()
            if not self._step_open:
                step_metrics.update(self.on_step_begin(metrics_step))
                self._step_open = True

            with self.accelerator.accumulate(self.model):
                output = self.train_forward_pass(batch)

                if output.metrics:
                    step_metrics.update(output.metrics)
                stats = self.backward_and_step(output.loss)
                if stats:
                    step_metrics.update(stats)

            if self.accelerator.sync_gradients:
                stats = self.after_optimizer_step(output.loss, metrics_step)
                if stats:
                    step_metrics.update(stats)
                pbar.update(1)

            self.metrics.batch_metrics(step_metrics, metrics_step)

    def after_optimizer_step(self, loss: torch.Tensor, step: int) -> MetricDict | None:
        step_metrics = MetricDict()

        if step % self.config.logging_steps == 0:
            step_metrics.add_metric(ScalarMetricEntry(tag="train/loss", scalar=loss.item()))
            if len(self.schedulers) > 1:
                for s, scheduler in enumerate(self.schedulers):
                    step_metrics.add_metric(
                        ScalarMetricEntry(tag=f"train/lr[{s}]", scalar=float(scheduler.get_last_lr()[0])))
            else:
                step_metrics.add_metric(
                    ScalarMetricEntry(tag="train/lr", scalar=float(self.schedulers[0].get_last_lr()[0])))

        self._step_open = False
        step_metrics.update(self.on_step_end(loss, step))

        if step % self.config.save_steps == 0:
            path = Path(self.metrics.log_dir).expanduser().resolve() / f"checkpoint-{step}"
            self.accelerator.save_state(str(path))
            self.save(step)

        if self.config.eval_steps is not None and step % self.config.eval_steps == 0:
            self.evaluate()
            self.model.train()

        self.train_state.global_step += 1
        self.train_state.step_in_epoch += 1

        return step_metrics

    @torch.no_grad()
    def evaluate(self) -> torch.Tensor:
        self.model.eval()
        # [loss_sum, count]

        eval_metrics = MetricDict()

        total = torch.zeros(2, device=self.accelerator.device)
        for batch in tqdm(self.eval_loader):
            output = self.eval_forward_pass(batch)
            loss = output.loss

            if output.metrics:
                eval_metrics.update(output.metrics)

            n_tokens = self.eval_token_count(batch)
            total += torch.stack([loss.float() * n_tokens, torch.tensor(float(n_tokens), device=total.device)])

        total = self.accelerator.reduce(total, reduction="sum")
        avg_loss = (total[0] / total[1])

        eval_metrics.add_metric(ScalarMetricEntry(tag="eval/loss", scalar=avg_loss.item()))

        self.metrics.batch_metrics(eval_metrics, self.train_state.global_step)

        return avg_loss

    def backward_and_step(self, loss: torch.Tensor) -> MetricDict | None:
        self.accelerator.backward(loss)
        grad_norm_metrics = self.clip_grads()

        [o.step() for o in self.optimizers]
        [s.step() for s in self.schedulers]
        [o.zero_grad() for o in self.optimizers]
        return grad_norm_metrics

    def clip_grads(self) -> MetricDict | None:
        stats: MetricDict | None = None
        if self.train_state.global_step % self.config.logging_steps == 0:
            self.metrics.add_grad_norms(self.model, self.train_state.global_step, per_param_scalars=True)

        if self.config.max_grad_norm is not None:
            total = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if total is not None and self.train_state.global_step % self.config.logging_steps == 0:
                total_norm = float(total)
                stats = MetricDict()
                stats.add_metrics([
                    ScalarMetricEntry(tag="grad_norm/total_unclipped", scalar=total_norm),
                    ScalarMetricEntry(
                        tag="grad_norm/clip_coef",
                        scalar=min(1.0, self.config.max_grad_norm / (total_norm + 1e-6))
                    ),
                ])
        return stats

    def save(self, step: int) -> None:
        path = Path(self.metrics.log_dir).expanduser().resolve() / f"checkpoint-{step}"
        self.accelerator.save_state(str(path))

    def on_step_begin(self, step: int) -> MetricDict:
        return self._callback_func(lambda c: c.on_step_begin, step)

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        return self._callback_func(lambda c: c.on_step_end, train_loss, step)

    def on_epoch_begin(self, epoch: int) -> MetricDict:
        return self._callback_func(lambda c: c.on_epoch_begin, epoch)

    def on_epoch_end(self, eval_loss: torch.Tensor | None, epoch: int) -> MetricDict:
        return self._callback_func(lambda c: c.on_epoch_end, eval_loss, epoch)

    def _callback_func[**P](
            self,
            callback_provider: Callable[[TrainingCallback], Callable[Concatenate[Self, P], MetricDict | None]],
            *args: P.args,
            **kwargs: P.kwargs
    ) -> MetricDict:
        step_metrics = MetricDict()
        for training_callback in self.training_callbacks:
            stats = callback_provider(training_callback)(self, *args, **kwargs)
            if stats is not None:
                step_metrics.update(stats)
        return step_metrics

    @abstractmethod
    def make_dataset_dict(self) -> Mapping[str, Dataset]:
        ...

    @abstractmethod
    def make_train_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        ...

    @abstractmethod
    def make_eval_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        ...

    @abstractmethod
    def make_criterion(self) -> criterion.Criterion:
        ...

    @abstractmethod
    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        ...

    @abstractmethod
    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        ...

    @abstractmethod
    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        ...

    def eval_token_count(self, batch: Mapping[str, Any]) -> int:
        return 1  # ctc loss is already token weighted; don't weight by tokens here for eval loss

    def train_batch_size(self, batch: Mapping[str, Any]) -> int:
        return self.config.train_batch_size

    def eval_batch_size(self, batch: Mapping[str, Any]) -> int:
        return self.config.eval_batch_size


class TrainState:
    def __init__(self) -> None:
        self.epoch = 0
        self.step_in_epoch = 0
        self.global_step = 0

    def state_dict(self) -> dict[str, Any]:
        return vars(self).copy()

    def load_state_dict(self, d: dict[str, Any]) -> None:
        vars(self).update(d)


class LarryTrainer[M: LarryModel, C: LarryTrainerConfig = LarryTrainerConfig](TrainerBase[M, C]):
    accelerator: Accelerator

    def make_dataset_dict(self) -> Mapping[str, Dataset]:
        return load_dataset(**self.config.dataset_config)

    def make_train_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        return DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=self.config.train_batch_size,
            drop_last=True,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
        )

    def make_eval_dataloader(self, dataset: Dataset | Mapping[str, Dataset]) -> DataLoader:
        return DataLoader(
            dataset["validation"],
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return None

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        return self._forward_pass(batch)

    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        return self._forward_pass(batch)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        try:
            model_output = self.model(**batch)
        except:
            self.log.error(
                f"Model forward failed with batch keys {batch.keys()} "
                f"and types {[type(v) for v in batch.values()]}"
            )
            raise
        try:
            loss = self.criterion(model_output, **batch)
        except:
            self.log.error(
                f"Criterion forward failed with type(model_outputs)={type(model_output)}, "
                f"batch.keys()={batch.keys()}, and "
                f"types={[type(v) for v in batch.values()]}"
            )
            raise
        return StepOutput(loss=loss, metrics=model_output.metrics)

    @abstractmethod
    def make_criterion(self) -> criterion.Criterion:
        ...
