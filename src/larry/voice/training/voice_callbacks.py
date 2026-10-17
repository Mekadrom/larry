import contextlib
import dataclasses
import random
from collections import defaultdict
from collections.abc import Generator
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.training import trainers
from larry.common.training.callbacks import TrainingCallback
from larry.voice.config.training.voice_trainer_configs import SpeakerLeakageProbesConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.training import voice_trainers

# TODO: make all modules/callbacks here agnostic of model type if possible (or take a generic interface)


class LinearSpeakerProbe(nn.Module):
    """Linear speaker probe; no activation functions, just a linear mapping."""

    def __init__(self, feature_dim: int, n_speakers: int) -> None:
        """Initializes the linear module."""
        super().__init__()
        self.probe = nn.Linear(feature_dim, n_speakers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Delegates to the linear module."""
        return self.probe(features)


class MLPSpeakerProbe(nn.Module):
    """Nonlinear speaker probe."""

    def __init__(self, feature_dim: int, inner_dim: int, n_speakers: int) -> None:
        """Initializes the simple MLP module."""
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(feature_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, n_speakers)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Delegates to the MLP module."""
        return self.probe(features)


class PooledActivations:
    """Collects the per-utterance mean and std over time of named modules' ``[B, T, D]`` outputs via forward hooks."""

    def __init__(self, layers: list[str]) -> None:
        """Takes the ``named_modules`` names whose outputs to pool."""
        self.layers = layers
        self.pooled: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.waveform_lengths = torch.empty(0, dtype=torch.long)
        self.waveform_steps = 1

    def begin_batch(self, waveform_lengths: torch.Tensor, waveform_steps: int) -> None:
        """Sets the valid lengths of the next forward's inputs, from which each layer's valid frames are scaled."""
        self.waveform_lengths = waveform_lengths
        self.waveform_steps = waveform_steps

    def hook(self, name: str, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        """Pools the first floating point tensor of a module's output over its valid frames."""
        tensor = output if isinstance(output, torch.Tensor) else next(
            o for o in output if isinstance(o, torch.Tensor) and o.is_floating_point()
        )
        if tensor.dim() != 3:
            raise ValueError(f"{name} output is {tuple(tensor.shape)}; probes need [B, T, D]")

        x = tensor.float()
        steps = x.shape[1]
        lengths = (self.waveform_lengths.to(x.device) * steps + self.waveform_steps - 1) // self.waveform_steps
        mask = (torch.arange(steps, device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        mean = (x * mask).sum(dim=1) / count
        std = (((x - mean[:, None]) ** 2 * mask).sum(dim=1) / count).sqrt()
        self.pooled[name].append(torch.cat([mean, std], dim=-1))

    @contextlib.contextmanager
    def attached(self, model: nn.Module) -> Generator[None]:
        """Registers the hooks for the duration of a ``with`` block, removing them however the block exits."""
        modules = dict(model.named_modules())
        missing = [n for n in self.layers if n not in modules]
        if missing:
            raise ValueError(f"no modules named {missing} in {type(model).__name__}")

        handles = [modules[n].register_forward_hook(partial(self.hook, n)) for n in self.layers]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def features(self) -> dict[str, torch.Tensor]:
        """Concatenates every pooled batch into one ``[N, 2D]`` tensor per layer, in forward order."""
        return {name: torch.cat(self.pooled[name]) for name in self.layers}


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """Accuracies of one probe at its early stopped weights, and the step those weights come from."""

    train_accuracy: float
    test_accuracy: float
    best_step: int


class SpeakerLeakageProbesCallback(TrainingCallback):
    """Trains a linear and non-linear probe on speaker identification on the specified modules' activations."""

    def __init__(self, config: SpeakerLeakageProbesConfig) -> None:
        """Initializes several tensors for caching train and eval input data."""
        super().__init__()
        self.config = config

        self.loader: DataLoader | None = None
        self.labels = torch.empty(0, dtype=torch.long)
        self.n_speakers = 0
        self.train_rows = torch.empty(0, dtype=torch.long)
        self.validation_rows = torch.empty(0, dtype=torch.long)
        self.test_rows = torch.empty(0, dtype=torch.long)

    def _initialize_from_eval_dataset(self, trainer: voice_trainers.SpeechEncoderTrainer) -> None:
        """Given a validation dataset, extracts train and test data for probe training and evaluation."""
        validation = cast(IndexedSpeechDataset, trainer.dataset_dict["validation"])
        rows, labels = self._select_rows(validation.speaker_keys())
        self.n_speakers = int(labels.max()) + 1 if len(rows) > 0 else 0

        _, durations = validation.source_ids_and_durations()
        batches = trainer.duration_batches(durations[rows])
        self.labels = labels[torch.tensor([p for batch in batches for p in batch], dtype=torch.long)]
        self.train_rows, self.validation_rows, self.test_rows = self._split_rows(self.labels)

        self.loader = DataLoader(
            Subset(validation, rows),
            batch_sampler=batches,
            collate_fn=trainer.make_collate_fn(),
            num_workers=trainer.config.num_dataloader_workers,
        )

    def _select_rows(self, keys: list[str]) -> tuple[list[int], torch.Tensor]:
        """Samples up to ``max_clips`` rows of every speaker with at least ``min_clips``, labeled with dense ids."""
        rows_by_speaker: dict[str, list[int]] = defaultdict(list)
        for row, key in enumerate(keys):
            rows_by_speaker[key].append(row)

        generator = random.Random(self.config.seed)
        rows: list[int] = []
        labels: list[int] = []
        eligible = sorted(k for k, r in rows_by_speaker.items() if len(r) >= self.config.min_clips)
        for label, key in enumerate(eligible):
            speaker_rows = rows_by_speaker[key]
            chosen = generator.sample(speaker_rows, min(self.config.max_clips, len(speaker_rows)))
            rows.extend(chosen)
            labels.extend([label] * len(chosen))

        return rows, torch.tensor(labels, dtype=torch.long)

    def _split_rows(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits row positions into probe train, validation, and test sets per speaker, so every speaker is in all."""
        generator = torch.Generator().manual_seed(self.config.seed)
        train: list[torch.Tensor] = [torch.empty(0, dtype=torch.long)]
        validation: list[torch.Tensor] = [torch.empty(0, dtype=torch.long)]
        test: list[torch.Tensor] = [torch.empty(0, dtype=torch.long)]
        for speaker in range(self.n_speakers):
            rows = torch.nonzero(labels == speaker).flatten()
            rows = rows[torch.randperm(len(rows), generator=generator)]
            n_test = max(1, round(len(rows) * self.config.test_fraction))
            n_validation = max(1, round(len(rows) * self.config.validation_fraction))
            test.append(rows[:n_test])
            validation.append(rows[n_test:n_test + n_validation])
            train.append(rows[n_test + n_validation:])

        return torch.cat(train), torch.cat(validation), torch.cat(test)

    def on_step_end(self, trainer: trainers.TrainerBase[Any, Any, Any], train_loss: torch.Tensor, step: int) -> MetricDict | None:
        """Every ``probe_every`` steps, trains speaker ID probes.

        Each configured module from the training model has its activations calculated for the train and eval probe
        datasets. Then, both a linear and a non-linear probe is trained on those activations and the evaluation of each
        probe determines the level of speaker identity leakage in the model at those layers.
        """
        if step % self.config.probe_every != 0 or not isinstance(trainer, voice_trainers.SpeechEncoderTrainer):
            return None

        if not trainer.accelerator.is_main_process:
            return None

        if self.loader is None:
            self._initialize_from_eval_dataset(trainer)

        if self.n_speakers < 2:
            return None

        metric_dict = MetricDict()
        metric_dict.add_metric(ScalarMetricEntry(tag="eval/speaker_probe/chance", scalar=1.0 / self.n_speakers))
        for layer, features in self._extract_features(trainer).items():
            for probe_name, result in self._probe_accuracies(features, trainer.accelerator.device).items():
                metric_dict.add_metric(ScalarMetricEntry(
                    tag=f"train/speaker_probe/{layer}/{probe_name}/accuracy", scalar=result.train_accuracy
                ))
                metric_dict.add_metric(ScalarMetricEntry(
                    tag=f"train/speaker_probe/{layer}/{probe_name}/best_step", scalar=result.best_step
                ))
                metric_dict.add_metric(ScalarMetricEntry(
                    tag=f"eval/speaker_probe/{layer}/{probe_name}/accuracy", scalar=result.test_accuracy
                ))

        return metric_dict

    def _extract_features(self, trainer: voice_trainers.SpeechEncoderTrainer) -> dict[str, torch.Tensor]:
        """Runs the probe clips through the frozen model once, pooling every configured layer's output."""
        model = trainer.accelerator.unwrap_model(trainer.model)
        device = trainer.accelerator.device
        activations = PooledActivations(self.config.layers)

        model_was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), activations.attached(model):
                for batch in cast(DataLoader, self.loader):
                    waveforms = batch["waveforms"].to(device)
                    waveform_lengths = batch["waveform_lengths"].to(device)
                    activations.begin_batch(waveform_lengths, waveforms.shape[-1])
                    model(waveforms, waveform_lengths)
        finally:
            model.train(model_was_training)

        return activations.features()

    def _make_probes(self, feature_dim: int) -> dict[str, nn.Module]:
        """Builds a fresh linear and MLP probe for features of the given width."""
        return {
            "linear": LinearSpeakerProbe(feature_dim, self.n_speakers),
            "mlp": MLPSpeakerProbe(feature_dim, self.config.mlp_inner_dim, self.n_speakers),
        }

    def _probe_accuracies(self, features: torch.Tensor, device: torch.device) -> dict[str, ProbeResult]:
        """Trains every probe on the probe train rows of one layer's features, early stopped on the validation rows."""
        features = features.to(device)
        train_x = features[self.train_rows.to(device)]
        validation_x = features[self.validation_rows.to(device)]
        test_x = features[self.test_rows.to(device)]
        mean = train_x.mean(dim=0)
        std = train_x.std(dim=0).clamp(min=1e-6)
        train_x = (train_x - mean) / std
        validation_x = (validation_x - mean) / std
        test_x = (test_x - mean) / std

        train_y = self.labels[self.train_rows].to(device)
        validation_y = self.labels[self.validation_rows].to(device)
        test_y = self.labels[self.test_rows].to(device)

        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [torch.cuda.current_device() if device.index is None else device.index]

        accuracies = {}
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.config.seed)
            for probe_name, probe in self._make_probes(features.shape[-1]).items():
                probe = probe.to(device)
                best_step = self._fit_probe(probe, train_x, train_y, validation_x, validation_y)
                accuracies[probe_name] = ProbeResult(
                    train_accuracy=self._accuracy(probe, train_x, train_y),
                    test_accuracy=self._accuracy(probe, test_x, test_y),
                    best_step=best_step,
                )

        return accuracies

    def _fit_probe(
            self,
            probe: nn.Module,
            features: torch.Tensor,
            labels: torch.Tensor,
            validation_features: torch.Tensor,
            validation_labels: torch.Tensor,
    ) -> int:
        """Full batch AdamW for up to ``train_epochs`` steps, keeping the weights that score best on validation.

        Returns the step the kept weights come from.
        """
        optimizer = torch.optim.AdamW(probe.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        best_accuracy = -1.0
        best_step = 0
        best_state: dict[str, torch.Tensor] = {}
        for step in range(1, self.config.train_epochs + 1):
            probe.train()
            loss = F.cross_entropy(probe(features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % self.config.validate_every == 0 or step == self.config.train_epochs:
                accuracy = self._accuracy(probe, validation_features, validation_labels)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_step = step
                    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}

        probe.load_state_dict(best_state)
        return best_step

    @staticmethod
    def _accuracy(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Top-1 accuracy of a probe."""
        probe.eval()
        with torch.no_grad():
            return (probe(features).argmax(dim=-1) == labels).float().mean().item()
