import dataclasses
import functools
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sized
from pathlib import Path
from typing import Any, ClassVar, Self, cast, Concatenate

import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from datasets import load_dataset
from torch import nn
from torch.optim import Optimizer, Muon, AdamW
from torch.optim.lr_scheduler import LRScheduler, ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.criteria import criterion
from larry.common.data.samplers import ResumableBatchSampler, ShuffledBatchSampler
from larry.common.metrics.metrics import MetricDict, Metrics, ScalarMetricEntry
from larry.common.model.model import LarryModelOutput
from larry.common.training.callbacks import TrainingCallback, ModelInitLoggingCallback
from larry.common.training.qk_clip import QKClipHandler, QKClipCallback
from larry.common.training.training_model import LarryModel
from larry.common.training.viz_callbacks import VizCallback
from larry.common.utils import git_utils
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


def _reduce_sum(accelerator: Accelerator, tensor: torch.Tensor) -> torch.Tensor:
    """Minor helper for casting the output of an accelerator tensor reduction to a tensor."""
    return cast(torch.Tensor, accelerator.reduce(tensor, reduction="sum"))


@dataclasses.dataclass(kw_only=True)
class StepOutput:
    """Wrapper dataclass for the output of a forward pass."""
    loss: torch.Tensor
    metrics: MetricDict
    sample_losses: torch.Tensor | None = None
    model_output: LarryModelOutput | None = None


class TrainerBase[M: LarryModel, C: LarryTrainerConfig = LarryTrainerConfig](Registrable, ABC, root=True):
    """Superclass for all trainers."""
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
        self.model_config = model.config

        self.training_callbacks = []

        dataset_dict = self.make_dataset_dict()
        train_loader = self.make_train_dataloader(dataset_dict)
        eval_loader = self.make_eval_dataloader(dataset_dict)

        self.update_provenance(self.provenance)

        if not isinstance(train_loader.batch_sampler, ResumableBatchSampler):
            raise TypeError(
                f"the train dataloader's batch_sampler must be a ResumableBatchSampler, "
                f"got {type(train_loader.batch_sampler).__name__}"
            )
        self.train_batch_sampler = train_loader.batch_sampler

        self.total_steps = self.config.max_steps

        optimizers = self.make_optimizers(model)  # e.g. [muon, adamw]
        schedulers = [self.make_lr_scheduler(o) for o in optimizers]

        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            step_scheduler_with_optimizer=False,
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
        self._pending_tokens = 0
        self.load(resume_dir)
        self.train_batch_sampler.start_at(
            self.train_state.global_step * self.config.gradient_accumulation * self.accelerator.num_processes
        )

        self.metrics = metrics

        self.add_default_training_callbacks()

        self.criterion = self.make_criterion()

        self._step_open = False

        self._compile_and_save_provenance()

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
        """Helper for building the kwargs to pass to any optimizer (Muon or AdamW agnostic)."""
        return dict(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

    def make_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Instantiates all (if any) LR schedulers.

        Has logic to break a configured warmup_steps into the originally configured LR scheduler prefaced with a linear
        LR schedule.
        """
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
        """Loads the train state from the specified ``resume_dir``."""
        self.accelerator.register_for_checkpointing(self.train_state)
        if resume_dir:
            self.accelerator.load_state(resume_dir)

    def count_tokens(self, n_tokens: int) -> None:
        """Allows tracking of total tokens seen by the trained model over time during training.

        Since most if not all trained models will train on variable-length sequence, this exists on the base class.
        """
        self._pending_tokens += n_tokens

    def sync_tokens_seen(self) -> int:
        """Reduces tokens seen across all devices' instances of this trainer class to use as the total tokens seen."""
        pending = torch.tensor(self._pending_tokens, dtype=torch.long, device=self.accelerator.device)
        self.train_state.tokens_seen += int(_reduce_sum(self.accelerator, pending).item())
        self._pending_tokens = 0
        return self.train_state.tokens_seen

    def add_default_training_callbacks(self) -> None:
        """Registers default callbacks for train step begin/end.

        Includes a QK clip callback automatically registered for when the configured optimizer is Muon.
        """
        if self.config.optimizer_name == "muon":
            self.training_callbacks.append(QKClipCallback(QKClipHandler(
                self.model,
                tau=self.config.muon_qk_clip_tau,
                probe_every=self.config.muon_qk_clip_probe_every,
                alpha=self.config.muon_qk_clip_alpha
            )))
        if self.config.viz_steps:
            self.training_callbacks.append(self.make_viz_callback(self.config.viz_steps))
        self.training_callbacks.append(ModelInitLoggingCallback())

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

        The train dataloader yields batches infinitely, so a break condition ends training.
        """
        self.model.train()

        for batch in self.train_loader:
            # ends training
            if self.train_state.global_step >= self.config.max_steps:
                break

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
                self.step_schedulers()
                stats = self.after_optimizer_step(output.loss, metrics_step)
                if stats:
                    step_metrics.update(stats)
                pbar.update(1)

            self.metrics.batch_metrics(step_metrics, metrics_step)

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

        self._step_open = False
        step_metrics.update(self.on_step_end(loss, step))

        self.train_state.global_step += 1
        completed = self.train_state.global_step

        if completed % self.config.save_steps == 0:
            self.save(completed)

        if self.config.eval_steps is not None and completed % self.config.eval_steps == 0:
            self.evaluate()
            self.model.train()

        return step_metrics

    @torch.no_grad()
    def evaluate(self) -> torch.Tensor:
        """Does the entire evaluation loop over the finite eval dataloader."""
        self.model.eval()
        # [loss_sum, count]

        eval_metrics = MetricDict()

        group_names = self.eval_group_names()
        groups = torch.zeros((len(group_names), 2), device=self.accelerator.device)
        statistics = self.eval_statistics()
        total = torch.zeros(2, device=self.accelerator.device)
        eval_batches = tqdm(
            self.eval_loader, desc="eval", leave=False, disable=not self.accelerator.is_local_main_process
        )
        for batch in eval_batches:
            output = self.eval_forward_pass(batch)
            loss = output.loss

            if output.metrics:
                eval_metrics.update(output.metrics)

            self.update_eval_statistics(statistics, batch, output)

            grouped = self.eval_group_totals(batch, output)
            if grouped is not None:
                group_ids, loss_sums, token_counts = grouped
                values = torch.stack([loss_sums.float(), token_counts.float()], dim=1).to(groups.device)
                groups.index_add_(0, group_ids.to(groups.device), values)
            else:
                n_tokens = self.eval_token_count(batch)
                total += torch.stack([loss.float() * n_tokens, torch.tensor(float(n_tokens), device=total.device)])

        if group_names:
            groups = _reduce_sum(self.accelerator, groups)
            for name, (loss_sum, token_count) in zip(group_names, groups.tolist()):
                if token_count > 0:
                    eval_metrics.add_metric(ScalarMetricEntry(tag=f"eval/loss/{name}", scalar=loss_sum / token_count))
            total = groups.sum(dim=0)
        else:
            total = _reduce_sum(self.accelerator, total)
        avg_loss = (total[0] / total[1])

        eval_metrics.add_metric(ScalarMetricEntry(tag="eval/loss", scalar=avg_loss.item()))
        eval_metrics.update(self.eval_statistics_metrics({
            name: _reduce_sum(self.accelerator, value)
            for name, value in statistics.items()
        }))

        self.metrics.batch_metrics(eval_metrics, self.train_state.global_step)

        return avg_loss

    def backward_and_step(self, loss: torch.Tensor) -> MetricDict | None:
        """Clips grads and performs all optimizer steps. Returns grad metrics from before zeroing."""
        self.accelerator.backward(loss)
        grad_norm_metrics = self.clip_grads()

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
        return self._callback_func(lambda c: c.on_step_begin, step)

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        """Calls all training callbacks through their on_step_end entrypoint."""
        return self._callback_func(lambda c: c.on_step_end, train_loss, step)

    def _callback_func[**P](
            self,
            callback_provider: Callable[[TrainingCallback], Callable[Concatenate[Self, P], MetricDict | None]],
            *args: P.args,
            **kwargs: P.kwargs
    ) -> MetricDict:
        """Helper wrapping the loop over training callbacks that collects metrics to return from each."""
        step_metrics = MetricDict()
        for training_callback in self.training_callbacks:
            stats = callback_provider(training_callback)(self, *args, **kwargs)
            if stats is not None:
                step_metrics.update(stats)
        return step_metrics

    @abstractmethod
    def make_dataset_dict(self) -> Mapping[str, Dataset]:
        """Allows for overriding to return the entire dataset."""
        ...

    @abstractmethod
    def make_train_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        """Allows for overriding to create the train dataloader."""
        ...

    @abstractmethod
    def make_eval_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
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

    def eval_group_names(self) -> list[str]:
        """Allows for overriding to return the list of eval dataset sources."""
        return []

    def eval_group_totals(
            self, batch: Mapping[str, Any], output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Allows for overriding to return the number of examples in a provided batch belonging to each source."""
        return None

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

    def eval_token_count(self, batch: Mapping[str, Any]) -> int:
        """Allows for overriding to return the number of tokens in a provided eval batch.

        Default implementation assumes the loss already reduces by token count.
        """
        return 1  # ctc loss is already token weighted; don't weight by tokens here for eval loss

    def train_batch_size(self, batch: Mapping[str, Any]) -> int:
        """Allows for overriding to determine the real batch size for a given train batch."""
        return self.config.train_batch_size

    def eval_batch_size(self, batch: Mapping[str, Any]) -> int:
        """Allows for overriding to determine the real batch size for a given eval batch."""
        return self.config.eval_batch_size

    def _compile_and_save_provenance(self) -> None:
        """Saves a separate json file that records the sources of this saved dataset."""

        self.provenance.populate_mutables()
        (Path(self.metrics.log_dir).expanduser().resolve() / "provenance.json").write_text(json.dumps(
            dataclasses.asdict(self.provenance),
            default=list,
            indent="\t"
        ))


class TrainState:
    """Dataclass for a trainer's training state. Allows resuming."""

    def __init__(self) -> None:
        self.global_step = 0
        self.tokens_seen = 0

    def state_dict(self) -> dict[str, Any]:
        return vars(self).copy()

    def load_state_dict(self, d: dict[str, Any]) -> None:
        vars(self).update(d)


class LarryTrainer[M: LarryModel, C: LarryTrainerConfig = LarryTrainerConfig](TrainerBase[M, C], ABC):
    """Implements a few abstract methods from ``TrainerBase`` with sensible defaults."""
    accelerator: Accelerator

    def make_dataset_dict(self) -> Mapping[str, Dataset]:
        """Uses datasets load_dataset with the ``TrainerConfig`` instance's ``dataset_config`` dict."""
        return load_dataset(**self.config.dataset_config)

    def make_train_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        """Creates a default simple train DataLoader for sharding."""
        train = dataset["train"]
        return DataLoader(
            train,
            batch_sampler=ShuffledBatchSampler(
                len(cast(Sized, train)), self.config.train_batch_size, self.config.seed
            ),
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
        )

    def make_eval_dataloader(self, dataset: Dataset | Mapping[str, Dataset]) -> DataLoader:
        """Creates a default simple eval DataLoader for sharding."""
        return DataLoader(
            dataset["validation"],
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """For simple dataloaders, allows overriding to create the collation function for train and eval."""
        return None

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Wraps a delegate forward pass function shared with eval."""
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
        """Forces implementation to create the criterion for subclasses of this base trainer class."""
        ...

    @abstractmethod
    def update_provenance(self, provenance: Provenance) -> None:
        """Forces implementation to create the criterion for subclasses of this base trainer class."""
        ...
