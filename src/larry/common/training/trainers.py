import contextlib
import dataclasses
import functools
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs, KwargsHandler
from datasets import Dataset as HFDataset
from torch import nn
from torch.optim import AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from larry.common.config.config import RunRecord
from larry.common.config.data.data_configs import DatasetSourceConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.criteria import criterion
from larry.common.criteria.criterion import LossTerm, weighted_total
from larry.common.data.dataloading.indexed_dataset import IndexedDataset, load_index, load_source
from larry.common.data.dataloading.samplers import ResumableBatchSampler
from larry.common.metrics.metrics import MetricDict, Metrics, ScalarMetricEntry
from larry.common.model.compile import ModelCompiler
from larry.common.model.flow import ConditionalFlowMatching
from larry.common.model.model import LarryModelOutput
from larry.common.training.callbacks import ModelInitLoggingCallback, TrainingCallback, TrainLossByTokens
from larry.common.training.qk_clip import QKClipCallback, QKClipHandler
from larry.common.training.training_model import LarryModel
from larry.common.training.viz_callbacks import VizCallback
from larry.common.utils import git_utils
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


def _reduce_sum(accelerator: Accelerator, tensor: torch.Tensor) -> torch.Tensor:
    """Minor helper for casting the output of an accelerator tensor reduction to a tensor."""
    return cast(torch.Tensor, accelerator.reduce(tensor, reduction="sum"))


MODEL_CONFIG_FILE = "config.json"


@dataclasses.dataclass(kw_only=True)
class StepOutput:
    """Wrapper dataclass for the output of a forward pass."""

    loss: torch.Tensor
    metrics: MetricDict
    terms: dict[str, LossTerm] | None = None
    model_output: LarryModelOutput | None = None


class TrainState:
    """Dataclass for a trainer's training state. Allows resuming."""

    def __init__(self) -> None:
        """Simple init."""
        self.global_step = 0
        self.tokens_seen = 0
        self.train_seconds = 0.0

    def state_dict(self) -> dict[str, Any]:
        """Converts to dict for saving."""
        return vars(self).copy()

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Converts from dict for loading."""
        vars(self).update(d)


class TrainerBase[
M: LarryModel,
C: LarryTrainerConfig[Any] = LarryTrainerConfig[Any],
D: Dataset = Dataset,
](Registrable, ABC, root=True, config_root=LarryTrainerConfig):
    """Superclass for all trainers."""

    REGISTRY: ClassVar[TypeRegistry[TrainerBase]]

    _model: M
    _config: C
    unwrapped_model: M

    train_state: TrainState

    training_callbacks: list[TrainingCallback]
    train_loader: DataLoaderShard
    eval_loader: DataLoaderShard
    optimizers: list[AcceleratedOptimizer]
    schedulers: list[AcceleratedScheduler]

    total_steps: int

    def __init__(
            self,
            provenance: Provenance,
            model: M,
            config: C,
            metrics: Metrics,
            resume_dir: str | None = None
    ) -> None:
        """Initializes any subclass of this abstract base class.

        Calls subclass overrides for creating the dataloaders, model, optimizers, scheduler(s), and callbacks, and uses
        accelerator to shard them across multiple gpus if applicable.
        """
        self.log = logging.getLogger(type(self).__name__)

        self.provenance = provenance
        self._config = config
        self.unwrapped_model = model
        self.model_config = model.config

        self.training_callbacks = []

        self.dataset_dict = self.make_dataset_dict()
        train_loader = self.make_train_dataloader(self.dataset_dict)
        eval_loader = self.make_eval_dataloader(self.dataset_dict)

        self.update_provenance(self.provenance)

        if not isinstance(train_loader.batch_sampler, ResumableBatchSampler):
            raise TypeError(
                f"the train dataloader's batch_sampler must be a ResumableBatchSampler, "
                f"got {type(train_loader.batch_sampler).__name__}"
            )
        self.train_batch_sampler = train_loader.batch_sampler

        self.total_steps = self.config.max_steps

        compiled = self.compile_model(model)
        if compiled:
            self.log.info(f"compiled {compiled} module forward(s) with {self.config.compile}")

        optimizers = self.make_optimizers(model)  # e.g. [muon, adamw]
        schedulers = [self.make_lr_scheduler(o) for o in optimizers]

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=self.kwargs_handlers(),
        )
        self._model, self.train_loader, self.eval_loader, *rest = self.accelerator.prepare(
            model,
            train_loader,
            eval_loader,
            *optimizers,
            *schedulers,
        )

        self.log.info(f"optimizers={optimizers}")
        self.log.info(f"schedulers={schedulers}")

        n_opt = len(optimizers)

        self.optimizers = cast(list[AcceleratedOptimizer], rest[:n_opt])
        self.schedulers = cast(list[AcceleratedScheduler], rest[n_opt:])

        self.log.info(f"self.optimizers={self.optimizers}")
        self.log.info(f"self.schedulers={self.schedulers}")

        self.train_state = TrainState()
        self._clock = time.monotonic()
        self._untimed = 0.0
        self._pending_tokens = 0
        self.load(resume_dir)
        self.train_batch_sampler.start_at(
            self.train_state.global_step * self.config.gradient_accumulation * self.accelerator.num_processes
        )

        self.metrics = metrics

        self.add_training_callbacks()

        self.criterion = self.make_criterion()

        self._step_open = False

        self._compile_and_save_provenance()
        self._save_model_config()

    @classmethod
    def prepare_model_config(cls, model_config: ModelConfig, config: C) -> ModelConfig:
        # lets a trainer derive the config it trains from the configured one, before the model is built
        return model_config

    def accelerator_kwargs_handlers(self) -> list[KwargsHandler]:
        return []

    def compile_model(self, model: nn.Module) -> int:
        return ModelCompiler(self.config.compile).apply(model)

    def kwargs_handlers(self) -> list[KwargsHandler]:
        handlers = self.accelerator_kwargs_handlers()
        if self.config.ddp_comm_hook == "no":
            return handlers

        hook = {
            "comm_hook": DDPCommunicationHookType(self.config.ddp_comm_hook),
            "comm_state_option": dict(self.config.ddp_comm_state_option),
        }
        others = [h for h in handlers if not isinstance(h, DistributedDataParallelKwargs)]
        ddp = [h for h in handlers if isinstance(h, DistributedDataParallelKwargs)]
        merged = dataclasses.replace(ddp[0], **hook) if ddp else DistributedDataParallelKwargs(**hook)
        return [*others, merged]

    def muon_excluded_names(self, model: nn.Module) -> list[str]:
        return self.config.muon_first_layer_names + self.config.muon_last_layer_names

    @property
    def model(self) -> M:
        """Immutable type-hint getter for the model instance."""
        return self._model

    @property
    def config(self) -> C:
        """Immutable type-hint getter for the config instance."""
        return self._config

    @functools.cached_property
    def n_parameters(self) -> int:
        """Calculates and returns the number of parameters for the model this trainer is training."""
        return sum(p.numel() for p in self.model.parameters())

    @functools.cached_property
    def git_commit_hash(self) -> str:
        """Returns the current git commit hash. Functionally equivalent to `git rev-parse HEAD`."""
        repo_state = git_utils.git_info()
        if repo_state is not None:
            return repo_state.git_commit
        return "head"

    @abstractmethod
    def update_provenance(self, provenance: Provenance) -> None:
        """Overrideable abstract method for updating provenance on start.

        Allows manual addition to datasets or models in the ancestry.
        """
        ...

    def make_optimizers(self, model: nn.Module) -> list[Optimizer]:
        """Instantiates one of two standard optimizers.

        Configuring for Muon still creates an AdamW optimizer for non-matrix parameters.
        """
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        if self.config.optimizer_name == "muon":
            excluded = set(self.muon_excluded_names(model))
            missing = excluded - {n for n, _ in model.named_parameters()}
            if missing:
                raise ValueError(f"excluded Muon params not found in model: {missing}")

            muon_params = []
            for n, p in trainable:
                if p.ndim == 2 and n not in excluded:
                    muon_params.append(p)

            other_params = []
            for n, p in trainable:
                if not (p.ndim == 2 and n not in excluded):
                    other_params.append(p)

            optimizers: list[Optimizer] = []
            if muon_params:
                optimizers.append(Muon(muon_params, **{
                    "adjust_lr_fn": "match_rms_adamw", **self._optimizer_kwargs(), **self.config.optimizer_config
                }))
            if other_params:
                optimizers.append(AdamW(other_params, betas=self.config.adamw_betas, **self._optimizer_kwargs()))
            return optimizers
        elif self.config.optimizer_name == "adamw":
            return [AdamW([p for _, p in trainable], **{
                "betas": self.config.adamw_betas, **self._optimizer_kwargs(), **self.config.optimizer_config
            })]
        raise ValueError(f"Unrecognized optimizer: {self.config.optimizer_name}")

    def _optimizer_kwargs(self) -> dict[str, Any]:
        """Helper for building the kwargs to pass to any optimizer (Muon or AdamW agnostic)."""
        return dict(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

    def lr_factor(self, step: int) -> float:
        """The LR multiplier at an optimizer step: linear warmup, then constant or cosine decay to ``min_lr``.

        Allows resuming - restores only the step count and the schedule comes from the current config.
        """
        config = self.config
        if step < config.warmup_steps:
            return 1e-8 + (1.0 - 1e-8) * step / config.warmup_steps

        if config.lr_scheduler_name == "constant":
            return 1.0
        if config.lr_scheduler_name != "cosine":
            raise ValueError(f"Unrecognized lr_scheduler_name: {config.lr_scheduler_name}")

        decay_start = config.warmup_steps if config.decay_start_step is None else config.decay_start_step
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, self.total_steps - decay_start))
        min_factor = config.min_lr / config.lr
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def make_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Instantiates the LR scheduler for one optimizer from ``lr_factor``."""
        # a plain function, not the bound method, so LambdaLR doesn't put its __dict__ into the saved state
        return LambdaLR(optimizer, lambda step: self.lr_factor(step))

    def load(self, resume_dir: str | None = None) -> None:
        """Loads the train state from the specified ``resume_dir``."""
        self.accelerator.register_for_checkpointing(self.train_state)
        if resume_dir:
            self.accelerator.load_state(resume_dir)

    def count_tokens(self, batch: Mapping[str, Any]) -> None:
        """Allows tracking of total tokens seen by the trained model over time during training.

        Since most if not all trained models will train on variable-length sequence, this exists on the base class.
        """
        ...

    def sync_tokens_seen(self) -> int:
        """Reduces tokens seen across all devices' instances of this trainer class to use as the total tokens seen."""
        pending = torch.tensor(self._pending_tokens, dtype=torch.long, device=self.accelerator.device)
        self.train_state.tokens_seen += int(_reduce_sum(self.accelerator, pending).item())
        self._pending_tokens = 0
        return self.train_state.tokens_seen

    def add_training_callbacks(self) -> None:
        """Registers default callbacks for train step begin/end.

        Includes a QK clip callback automatically registered for when the configured optimizer is Muon.
        """
        if self.config.optimizer_name == "muon":
            self.training_callbacks.append(QKClipCallback(
                self,
                QKClipHandler(
                    self.model,
                    tau=self.config.muon_qk_clip_tau,
                    probe_every=self.config.muon_qk_clip_probe_every,
                    alpha=self.config.muon_qk_clip_alpha
                )
            ))
        if self.config.viz_steps:
            self.training_callbacks.append(self.make_viz_callback(self.config.viz_steps))
        self.training_callbacks.append(ModelInitLoggingCallback(self))
        self.training_callbacks.append(TrainLossByTokens(self, self.config.logging_steps))

    def run(self) -> None:
        """Begins a training run.

        Does not iterate in "epochs", it iterates an infinite dataloader for the configured number of steps.
        """
        if self.train_state.global_step == 0 and self.config.eval_steps is not None:
            self.evaluate()

        with tqdm(
                initial=self.train_state.global_step,
                total=self.config.max_steps,
                desc="train",
                disable=not self.accelerator.is_local_main_process
        ) as pbar:
            self.train(pbar=pbar)

        # force saves the model if the callback for end of step didn't end up saving on the last step
        if self.train_state.global_step % self.config.save_steps != 0:
            self.save(self.train_state.global_step)

    def train(self, pbar: tqdm) -> None:
        """Does the entire training loop for all configured steps.

        The train dataloader yields batches infinitely, so the step count ends training before the next batch is fetched.
        """
        self.model.train()
        self._clock = time.monotonic()
        self._untimed = 0.0

        batches = iter(self.train_loader)
        window_losses: list[torch.Tensor] = []
        window_terms: dict[str, list[torch.Tensor]] = {}
        while self.train_state.global_step < self.config.max_steps:
            batch = next(batches)

            metrics_step = self.train_state.global_step

            step_metrics = MetricDict()
            if not self._step_open:
                step_metrics.update(self.on_step_begin(metrics_step))
                self._step_open = True

            with self.accelerator.accumulate(self.model):
                output = self.train_forward_pass(batch)
                window_losses.append(output.loss.detach())
                for name, term in (output.terms or {}).items():
                    window_terms.setdefault(name, []).append(term.value.detach())

                if output.metrics:
                    step_metrics.update(output.metrics)
                stats = self.backward_and_step(output.loss)
                if stats:
                    step_metrics.update(stats)

            if self.accelerator.sync_gradients:
                self.step_schedulers()
                window_loss = torch.stack(window_losses).float().mean()
                window_losses.clear()
                stats = self.after_optimizer_step(window_loss, metrics_step)
                if stats:
                    step_metrics.update(stats)

                if metrics_step % self.config.logging_steps == 0:
                    step_metrics.update(self.term_metrics("train", {
                        name: torch.stack(values).float().mean() for name, values in window_terms.items()
                    }))

                window_terms.clear()
                pbar.update(1)

            self.metrics.batch_metrics(step_metrics, metrics_step)

    @contextlib.contextmanager
    def untimed(self) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self._untimed += time.monotonic() - start

    def tick_train_clock(self) -> None:
        now = time.monotonic()
        self.train_state.train_seconds += max(0.0, now - self._clock - self._untimed)
        self._clock = now
        self._untimed = 0.0

    def after_optimizer_step(self, loss: torch.Tensor, step: int) -> MetricDict | None:
        """Handles callbacks, metrics, and recordkeeping."""
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
            step_metrics.add_metric(
                ScalarMetricEntry(tag="time/train_hours", scalar=self.train_state.train_seconds / 3600))

        self._step_open = False
        step_metrics.update(self.on_step_end(loss, step))

        self.train_state.global_step += 1
        completed = self.train_state.global_step
        self.tick_train_clock()

        if completed % self.config.save_steps == 0:
            with self.untimed():
                self.save(completed)

        if self.config.eval_steps is not None and completed % self.config.eval_steps == 0:
            with self.untimed():
                self.evaluate()
                self.model.train()

        return step_metrics

    @staticmethod
    def term_metrics(split: str, values: Mapping[str, torch.Tensor], suffix: str = "") -> MetricDict:
        """``{split}/{term}{suffix}`` for every component; the weighted total is logged separately as the loss."""
        metrics = MetricDict()
        for name, value in values.items():
            if name != "loss":
                metrics.add_metric(ScalarMetricEntry(tag=f"{split}/{name}{suffix}", scalar=float(value)))

        return metrics

    def flow_matchers(self) -> list[ConditionalFlowMatching]:
        """Every flow-matching path in the model, whether the module is one or holds one."""
        found: list[ConditionalFlowMatching] = []
        for module in self.accelerator.unwrap_model(self.model).modules():
            if isinstance(module, ConditionalFlowMatching):
                found.append(module)

            found += [value for value in vars(module).values() if isinstance(value, ConditionalFlowMatching)]

        return found

    @contextlib.contextmanager
    def fixed_eval_noise(self) -> Iterator[None]:
        """Every eval draws the same flow-matching t and noise for the same rows, so flow losses compare across runs."""
        flows = self.flow_matchers()
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.eval_noise_seed)
        for flow in flows:
            flow.noise_generator = generator

        try:
            yield
        finally:
            for flow in flows:
                flow.noise_generator = None

    @torch.no_grad()
    def evaluate(self) -> torch.Tensor:
        """Does the entire evaluation loop over the finite eval dataloader.

        Every loss term is summed per row, so the loss and each of its components are logged overall and for every
        grouping of rows (per source, plus whatever ``eval_groupings`` adds).
        """
        self.model.eval()

        eval_metrics = MetricDict()
        groupings = self.eval_groupings()
        statistics = self.eval_statistics()
        device = self.accelerator.device
        weights: dict[str, float] = {}
        # [term] -> [sum, count] overall, and [term] -> (n_groups, 2) for each grouping
        overall: dict[str, torch.Tensor] = {}
        grouped: list[dict[str, torch.Tensor]] = [{} for _ in groupings]

        eval_batches = tqdm(
            self.eval_loader,
            desc="eval",
            leave=False,
            disable=not self.accelerator.is_local_main_process,
        )
        with self.fixed_eval_noise():
            for batch in eval_batches:
                output = self.eval_forward_pass(batch)
                if output.metrics:
                    eval_metrics.update(output.metrics)

                self.update_eval_statistics(statistics, batch, output)

                terms = output.terms or {
                    "loss": LossTerm(weight=1.0, sums=output.loss.reshape(1), counts=torch.ones(1, device=device))
                }

                group_ids = self.eval_group_ids(batch)
                for name, term in terms.items():
                    weights[name] = term.weight
                    row_values = torch.stack([term.sums.float(), term.counts.float()], dim=1).to(device)
                    overall[name] = overall.get(name, torch.zeros(2, device=device)) + row_values.sum(dim=0)
                    # terms without one value per row (a criterion that only reports a batch loss) can't be grouped
                    if group_ids is None or row_values.shape[0] != group_ids[0].shape[0]:
                        continue

                    for (_, names), ids, accumulated in zip(groupings, group_ids, grouped, strict=True):
                        totals = accumulated.setdefault(name, torch.zeros(len(names), 2, device=device))
                        totals.index_add_(0, ids.to(device), row_values)

        overall = {name: _reduce_sum(self.accelerator, values) for name, values in overall.items()}
        avg_loss = self.log_eval_terms(eval_metrics, overall, weights, "")
        for (prefix, names), accumulated in zip(groupings, grouped, strict=True):
            accumulated = {name: _reduce_sum(self.accelerator, values) for name, values in accumulated.items()}
            for g, group_name in enumerate(names):
                self.log_eval_terms(
                    eval_metrics,
                    {
                        name: values[g]
                        for name, values in accumulated.items()
                    },
                    weights,
                    f"/{prefix}{group_name}",
                )

        eval_metrics.update(self.eval_statistics_metrics({
            name: _reduce_sum(self.accelerator, value)
            for name, value in statistics.items()
        }))

        self.metrics.batch_metrics(eval_metrics, self.train_state.global_step)

        return avg_loss

    def log_eval_terms(
            self,
            metrics: MetricDict,
            totals: Mapping[str, torch.Tensor],
            weights: Mapping[str, float],
            suffix: str,
    ) -> torch.Tensor:
        """Logs the weighted total and every component for one group of rows; skips terms the group never had."""
        values = {name: sums_counts[0] / sums_counts[1] for name, sums_counts in totals.items() if sums_counts[1] > 0}
        if not values:
            return torch.tensor(float("nan"))

        total = torch.stack([weights[name] * value for name, value in values.items()]).sum()
        metrics.add_metric(ScalarMetricEntry(tag=f"eval/loss{suffix}", scalar=total.item()))
        metrics.update(self.term_metrics("eval", values, suffix))
        return total

    def backward_and_step(self, loss: torch.Tensor) -> MetricDict | None:
        """Clips grads and performs all optimizer steps. Returns grad metrics from before zeroing."""
        self.accelerator.backward(loss)
        grad_norm_metrics = self.clip_grads() if self.accelerator.sync_gradients else None

        [o.step() for o in self.optimizers]
        [o.zero_grad() for o in self.optimizers]
        return grad_norm_metrics

    def step_schedulers(self) -> None:
        """Advances every LR scheduler once per optimizer step, whatever the number of processes."""
        if any(o.step_was_skipped for o in self.optimizers):
            return
        [s.step() for s in self.schedulers]

    def clip_grads(self) -> MetricDict | None:
        """Performs grad norm clipping, returns metrics related to grads."""
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
        """Saves the model and train state to file as a checkpoint."""
        self.sync_tokens_seen()
        path = Path(self.metrics.log_dir).expanduser().resolve() / f"checkpoint-{step}"
        self.accelerator.save_state(str(path))

    def on_step_begin(self, step: int) -> MetricDict:
        """Calls all training callbacks through their on_step_begin entrypoint."""
        step_metrics = MetricDict()
        for training_callback in self.training_callbacks:
            stats = training_callback.on_step_begin(step)
            if stats is not None:
                step_metrics.update(stats)
        return step_metrics

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        """Calls all training callbacks through their on_step_end entrypoint."""
        step_metrics = MetricDict()
        for training_callback in self.training_callbacks:
            with contextlib.nullcontext() if training_callback.counts_as_training else self.untimed():
                stats = training_callback.on_step_end(train_loss, step)
            if stats is not None:
                step_metrics.update(stats)
        return step_metrics

    @abstractmethod
    def make_dataset_dict(self) -> Mapping[str, D]:
        """Allows for overriding to return the entire dataset."""
        ...

    @abstractmethod
    def make_train_dataloader(self, dataset: Mapping[str, D]) -> DataLoader:
        """Allows for overriding to create the train dataloader."""
        ...

    @abstractmethod
    def make_eval_dataloader(self, dataset: Mapping[str, D]) -> DataLoader:
        """Allows for overriding to create the eval dataloader."""
        ...

    @abstractmethod
    def make_criterion(self) -> criterion.Criterion:
        """Allows for overriding to create the criterion for training."""
        ...

    @abstractmethod
    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        """Allows for overriding to create the default visualization training callback."""
        ...

    @abstractmethod
    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Allows for overriding to implement the real train forward pass for the training model."""
        ...

    @abstractmethod
    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Allows for overriding to implement the real eval forward pass for the training model."""
        ...

    def eval_groupings(self) -> list[tuple[str, list[str]]]:
        """(tag prefix, group names) for every way eval rows are grouped; by default, one group per eval source."""
        return [("", [source.name for source in self.config.mixture_config.eval_sources])]

    def eval_group_ids(self, batch: Mapping[str, Any]) -> list[torch.Tensor] | None:
        """Each row's group under every grouping, in ``eval_groupings`` order; None when the batch can't be grouped."""
        if "source_ids" not in batch:
            return None

        return [batch["source_ids"]]

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        """Allows for overriding to initialize metrics to log under eval."""
        return {}

    def update_eval_statistics(
            self, statistics: dict[str, torch.Tensor], batch: Mapping[str, Any], output: StepOutput
    ) -> None:
        """Allows for overriding to update the empty metrics created by ``eval_statistics``"""
        ...

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        """Collects eval statistics and converts them to a ``MetricDict`` for logging."""
        return MetricDict()

    def train_batch_size(self, batch: Mapping[str, Any]) -> int:
        """Allows for overriding to determine the real batch size for a given train batch."""
        return self.config.train_batch_size

    def eval_batch_size(self, batch: Mapping[str, Any]) -> int:
        """Allows for overriding to determine the real batch size for a given eval batch."""
        return self.config.eval_batch_size

    def _save_model_config(self) -> None:
        if not self.accelerator.is_main_process:
            return

        path = Path(self.metrics.log_dir).expanduser().resolve() / MODEL_CONFIG_FILE
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.accelerator.unwrap_model(self.model).config.to_dict(), indent=2))

    def _compile_and_save_provenance(self) -> None:
        """Saves a separate json file that records the sources of this saved dataset."""
        self.provenance.populate_mutables()
        run_dir = Path(self.metrics.log_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "provenance.json").write_text(json.dumps(self.provenance.to_dict(), indent="\t"))
        if self.provenance.config is not None and self.provenance.cli is not None:
            RunRecord(config=self.provenance.config, cli=self.provenance.cli).write(run_dir)


class LarryTrainer[
M: LarryModel,
C: LarryTrainerConfig[Any] = LarryTrainerConfig[Any],
S: DatasetSourceConfig = DatasetSourceConfig,
D: IndexedDataset[Any] = IndexedDataset[Any],
](TrainerBase[M, C, D], ABC):
    """Implements a few abstract methods from ``TrainerBase`` with sensible defaults."""

    accelerator: Accelerator

    def make_dataset_dict(self) -> Mapping[str, D]:
        """Loads the indexed dataset sources from config for this ``SpeechTokenizerTrainer``."""
        if len(self.config.mixture_config.train_sources) == 0:
            raise ValueError("train_sources is empty")

        if len(self.config.mixture_config.eval_sources) == 0:
            raise ValueError("eval_sources is empty")

        mixture = self.config.mixture_config
        return {
            "train": self._load_indexed(mixture.train_sources),
            "validation": self._load_indexed(mixture.eval_sources, is_eval=True),
        }

    def _load_indexed(self, configs: list[S], is_eval: bool = False) -> D:
        """Helper for loading the indexed dataset source configuration. Index datasets can be materialized."""
        indexes: list[HFDataset] = []
        sources: list[HFDataset | None] = []
        for config in configs:
            index = load_index(config)
            source = load_source(config)
            if source is None:
                index = self.transform_as_source(config, index)
                sources.append(None)
            else:
                sources.append(self.transform_as_source(config, source))
            indexes.append(index)
        return self.load_indexed(configs, indexes, sources, is_eval=is_eval)

    def transform_as_source(self, config: S, dataset: HFDataset) -> HFDataset:
        return dataset

    @abstractmethod
    def load_indexed(
            self,
            configs: list[S],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            is_eval: bool = False,
    ) -> D:
        ...

    def make_train_dataloader(self, dataset: Mapping[str, D]) -> DataLoader:
        """Loads the weighted dataloader for the train dataset."""
        train = dataset["train"]
        weights = [
            source.weight
            for source in self.config.mixture_config.train_sources
        ]
        sampler = self.make_train_sampler(train, weights)
        return DataLoader(
            train,
            batch_sampler=sampler,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
        )

    def make_eval_dataloader(self, dataset: Dataset | Mapping[str, D]) -> DataLoader:
        """Loads the weighted dataloader for the validation dataset."""
        validation = dataset["validation"]
        return DataLoader(
            validation,
            batch_sampler=self.make_eval_sampler(validation),
            collate_fn=self.make_eval_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """For simple dataloaders, allows overriding to create the collation function for train and eval."""
        return None

    def make_eval_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return self.make_collate_fn()

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Wraps _forward_pass to first keep a running count of total tokens seen by the model in training."""
        self.count_tokens(batch)
        return self._forward_pass(batch)

    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Wraps a delegate forward pass function shared with train."""
        return self._forward_pass(batch)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Shared forward pass logic between train and eval.

        Does a model forward pass followed by criterion forward pass.
        """
        try:
            model_output = self.model(**batch)
        except:
            self.log.error(
                f"Model forward failed with batch keys {batch.keys()} "
                f"and types {[type(v) for v in batch.values()]}"
            )
            raise

        try:
            terms = self.criterion.loss_terms(model_output, **batch)
        except:
            self.log.error(
                f"Criterion forward failed with type(model_outputs)={type(model_output)}, "
                f"batch.keys()={batch.keys()}, and "
                f"types={[type(v) for v in batch.values()]}"
            )
            raise

        return self.step_output(model_output, terms)

    @staticmethod
    def step_output(model_output: LarryModelOutput, terms: dict[str, LossTerm]) -> StepOutput:
        """The loss is the weighted total of the criterion's terms; the terms ride along for logging."""
        return StepOutput(
            loss=weighted_total(terms),
            metrics=model_output.metrics or MetricDict(),
            terms=terms,
            model_output=model_output,
        )

    def update_provenance(self, provenance: Provenance) -> None:
        for source in self.config.mixture_config.train_sources:
            provenance.ancestry.datasets.add(source.provenance_name or source.index_path)

    @abstractmethod
    def make_train_sampler(self, dataset: D, weights: list[float]) -> Sampler:
        ...

    @abstractmethod
    def make_eval_sampler(self, dataset: D) -> Sampler | Iterable:
        ...

    @abstractmethod
    def make_criterion(self) -> criterion.Criterion:
        """Forces implementation to create the criterion for subclasses of this base trainer class."""
        ...
