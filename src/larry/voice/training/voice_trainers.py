import functools
from pathlib import Path
from abc import ABC
from collections.abc import Mapping, Callable
from typing import Any, cast

import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader

from larry.common.config.criteria.criterion_configs import CTCCriterionConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.criteria.criterion import Criterion
from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.model.model import LarrySpeechTokenizerModelOutput
from larry.common.training.trainers import LarryTrainer, VizCallback, StepOutput
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer, TextTokenizer
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.config.training.voice_trainer_configs import VoiceTrainerConfig, SpeechTokenizerTrainerConfig
from larry.voice.criteria.ctc import CTCCriterion
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.data.dataloading.voice_dataloading import WeightedBucketBatchSampler, SpeechCTCCollator, \
    SpeechSourceConfig, index_split_files
from larry.voice.model.fsq import code_counts, codebook_statistics
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.model.voice_models import VoiceModel
from larry.voice.training.voice_viz_callbacks import SpeechTokenizerVizCallback


def load_index(config: SpeechSourceConfig) -> HFDataset:
    if Path(config.index_path).expanduser().is_dir():
        return load_dataset("parquet", data_files=index_split_files(config.index_path, config.split), split="train")
    return load_dataset(config.index_path, split=config.split)


class VoiceTrainer[M: VoiceModel, C: VoiceTrainerConfig = VoiceTrainerConfig](LarryTrainer[M, C], ABC):
    ...


class SpeechTokenizerTrainer(VoiceTrainer[SpeechTokenizerModel, SpeechTokenizerTrainerConfig]):
    model_config: SpeechTokenizerModelConfig

    @functools.cached_property
    def tokenizer(self) -> TextTokenizer:
        return SentencePieceTokenizer(self.config.tokenizer_model_path)

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        self.count_tokens(batch["n_target_tokens"])
        return self._forward_pass(batch)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self.model(batch["waveforms"], batch["waveform_lengths"])
        loss = self.criterion(output, targets=batch["targets"], target_lengths=batch["target_lengths"])
        metrics = MetricDict(output.metrics or {})
        if self.train_state.global_step % self.config.logging_steps == 0:
            counts = code_counts(output.token_ids, output.frame_lengths, self.codebook_size)[0]
            statistics = codebook_statistics(counts, self.model_config.fsq_dims, self.model_config.fsq_levels)
            for name, value in statistics.items():
                if not name.startswith("dim"):
                    metrics.add_metric(ScalarMetricEntry(tag=f"train/fsq/{name}", scalar=value))
        return StepOutput(loss=loss, metrics=metrics, model_output=output)

    @property
    def codebook_size(self) -> int:
        return self.model_config.fsq_levels ** self.model_config.fsq_dims

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        if step % self.config.logging_steps == 0:
            self.metrics.add_scalar("train/loss_by_tokens", train_loss.item(), self.sync_tokens_seen())
        return self._callback_func(lambda c: c.on_step_end, train_loss, step)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return SpeechTokenizerVizCallback(viz_steps)

    def make_criterion(self) -> Criterion:
        if self.tokenizer.vocab_size() != self.model_config.text_vocab_size:
            raise ValueError(
                f"vocab_sizes do not match: "
                f"tokenizer={self.tokenizer.vocab_size()}, model_config={self.model_config.text_vocab_size}"
            )
        return CTCCriterion(CTCCriterionConfig(blank=self.model_config.text_vocab_size), self.model)

    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self.model(batch["waveforms"], batch["waveform_lengths"])
        criterion = cast(CTCCriterion, self.criterion)
        sample_losses = criterion.sample_losses(output, batch["targets"], batch["target_lengths"])
        loss = (sample_losses / batch["target_lengths"].clamp(min=1)).mean()
        return StepOutput(loss=loss, metrics=output.metrics, sample_losses=sample_losses, model_output=output)

    def eval_group_names(self) -> list[str]:
        names = [source.name for source in self.config.mixture_config.eval_sources]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"eval source names must be unique to log per-source eval loss: {duplicates}")
        return names

    def eval_group_totals(
            self, batch: Mapping[str, Any], output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if output.sample_losses is None:
            return None
        return batch["source_ids"], output.sample_losses, batch["target_lengths"]

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        n_sources = len(self.config.mixture_config.eval_sources)
        counts = torch.zeros((n_sources, self.codebook_size), dtype=torch.long, device=self.accelerator.device)
        return {"fsq_counts": counts}

    def update_eval_statistics(
            self, statistics: dict[str, torch.Tensor], batch: Mapping[str, Any], output: StepOutput
    ) -> None:
        model_output = cast(LarrySpeechTokenizerModelOutput, output.model_output)
        statistics["fsq_counts"] += code_counts(
            model_output.token_ids,
            model_output.frame_lengths,
            self.codebook_size,
            batch["source_ids"],
            len(self.config.mixture_config.eval_sources),
        ).to(statistics["fsq_counts"].device)

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        metrics = MetricDict()
        counts = statistics["fsq_counts"]
        dims, levels = self.model_config.fsq_dims, self.model_config.fsq_levels
        for name, value in codebook_statistics(counts.sum(dim=0), dims, levels).items():
            metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}", scalar=value))
        for source, source_counts in zip(self.config.mixture_config.eval_sources, counts):
            for name, value in codebook_statistics(source_counts, dims, levels).items():
                if not name.startswith("dim"):
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}/{source.name}", scalar=value))
        return metrics

    def make_dataset_dict(self) -> Mapping[str, Dataset]:
        if len(self.config.mixture_config.train_sources) == 0:
            raise ValueError("train_sources is empty")
        if len(self.config.mixture_config.eval_sources) == 0:
            raise ValueError("eval_sources is empty")
        mixture = self.config.mixture_config
        return {
            "train": self._load_indexed(mixture.train_sources),
            "validation": self._load_indexed(mixture.eval_sources),
        }

    def _load_indexed(self, configs: list[SpeechSourceConfig]) -> IndexedSpeechDataset:
        indexes: list[HFDataset] = []
        sources: list[HFDataset | None] = []
        for config in configs:
            index = load_index(config)
            if config.source_path is None:
                index = index.cast_column(config.audio_column, Audio(decode=False))
                sources.append(None)
            else:
                source = load_dataset(
                    config.source_path,
                    name=config.source_name,
                    split=config.source_split,
                    revision=config.source_revision,
                )
                sources.append(source.cast_column(config.audio_column, Audio(decode=False)))
            indexes.append(index)
        return IndexedSpeechDataset(configs, indexes, sources, self.config.mixture_config.sample_rate)

    def make_train_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        train = cast(IndexedSpeechDataset, dataset["train"])
        source_ids, durations = train.source_ids_and_durations()
        weights = [
            source.weight
            for source in self.config.mixture_config.train_sources
        ]
        sampler = WeightedBucketBatchSampler(
            source_ids,
            durations,
            weights,
            max_batch_seconds=self.config.mixture_config.max_batch_seconds,
            pool_size=self.config.mixture_config.pool_size,
            seed=self.config.mixture_config.seed,
        )
        return DataLoader(
            train,
            batch_sampler=sampler,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
        )

    def make_eval_dataloader(self, dataset: Dataset | Mapping[str, Dataset]) -> DataLoader:
        # noinspection PyUnnecessaryCast
        validation = cast(IndexedSpeechDataset, dataset["validation"])
        _, durations = validation.source_ids_and_durations()
        batches: list[list[int]] = []
        batch: list[int] = []
        for row in torch.argsort(durations).tolist():
            padded_seconds = (len(batch) + 1) * float(durations[row])
            if len(batch) > 0 and padded_seconds > self.config.mixture_config.max_batch_seconds:
                batches.append(batch)
                batch = []
            batch.append(row)
        if len(batch) > 0:
            batches.append(batch)
        return DataLoader(
            validation,
            batch_sampler=batches,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return SpeechCTCCollator(self.tokenizer)

    def update_provenance(self, provenance: Provenance) -> None:
        for source in self.config.mixture_config.train_sources:
            provenance.ancestry.datasets.add(source.source_path or source.index_path)
