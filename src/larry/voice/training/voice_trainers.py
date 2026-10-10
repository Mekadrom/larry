import functools
from abc import ABC
from collections.abc import Mapping, Callable
from typing import Any, cast

import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader

from larry.common.config.criteria.criterion_configs import CTCCriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.metrics.metrics import Metrics, AggregateMetricEntry, MetricDict
from larry.common.training.trainers import LarryTrainer, VizCallback, StepOutput
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer, TextTokenizer
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.config.training.voice_trainer_configs import VoiceTrainerConfig, SpeechTokenizerTrainerConfig
from larry.voice.criteria.ctc import CTCCriterion
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.data.dataloading.voice_dataloading import WeightedBucketBatchSampler, SpeechCTCCollator, \
    SpeechSourceConfig
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.model.voice_models import VoiceModel
from larry.voice.training.voice_viz_callbacks import SpeechTokenizerVizCallback


class VoiceTrainer[M: VoiceModel, C: VoiceTrainerConfig = VoiceTrainerConfig](LarryTrainer[M, C], ABC):
    ...


class SpeechTokenizerTrainer(VoiceTrainer[SpeechTokenizerModel, SpeechTokenizerTrainerConfig]):
    model_config: SpeechTokenizerModelConfig

    def __init__(
            self,
            model: SpeechTokenizerModel,
            config: SpeechTokenizerTrainerConfig,
            metrics: Metrics,
            resume_dir: str | None = None
    ) -> None:
        self.train_sampler: WeightedBucketBatchSampler | None = None
        super().__init__(model, config, metrics, resume_dir=resume_dir)
        self.tokens_seen = AggregateMetricEntry(tag="train/total_tokens", average_reduce="none")

    @functools.cached_property
    def tokenizer(self) -> TextTokenizer:
        return SentencePieceTokenizer(self.config.tokenizer_model_path)

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        self.tokens_seen.add_(batch["target_lengths"].sum().item())
        return self._forward_pass(batch)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self.model(batch["waveforms"], batch["waveform_lengths"])
        loss = self.criterion(output, targets=batch["targets"], target_lengths=batch["target_lengths"])
        return StepOutput(loss=loss, metrics=output.metrics)

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        if step % self.config.logging_steps == 0:
            self.metrics.add_scalar("train/loss_by_tokens", train_loss.item(), int(self.tokens_seen.value))
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

    def eval_token_count(self, batch: Mapping[str, Any]) -> int:
        return int(batch["target_lengths"].sum().item())

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
            index = load_dataset(config.index_path, split=config.split)
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
        self.train_sampler = WeightedBucketBatchSampler(
            source_ids,
            durations,
            weights,
            max_batch_seconds=self.config.mixture_config.max_batch_seconds,
            pool_size=self.config.mixture_config.pool_size,
            seed=self.config.mixture_config.seed,
        )
        return DataLoader(
            train,
            batch_sampler=self.train_sampler,
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
        )

    def make_eval_dataloader(self, dataset: Dataset | Mapping[str, Dataset]) -> DataLoader:
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

    def on_epoch_begin(self, epoch: int) -> MetricDict:
        # don't rely on accelerate forwarding set_epoch through its batch-sampler wrapper
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        return super().on_epoch_begin(epoch)
