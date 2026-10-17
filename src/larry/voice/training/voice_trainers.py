import dataclasses
import functools
import json
from abc import ABC
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset
from huggingface_hub import hf_hub_download
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader, Dataset
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from larry.common.config.criteria.criterion_configs import CTCCriterionConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.criteria.criterion import Criterion
from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.model.model import LarrySpeechTokenizerModelOutput
from larry.common.training.trainers import LarryTrainer, StepOutput, VizCallback
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer, TextTokenizer
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.config.preprocessing.voice_configs import SpeechSourceConfig
from larry.voice.config.training.voice_trainer_configs import SpeechTokenizerTrainerConfig, VoiceTrainerConfig
from larry.voice.criteria.ctc import CTCCriterion
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.data.dataloading.voice_dataloading import (
    SpeechCTCCollator,
    WeightedBucketBatchSampler,
    index_split_files,
)
from larry.voice.model.fsq import code_counts, codebook_statistics
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.model.voice_models import VoiceModel
from larry.voice.training import voice_callbacks
from larry.voice.training.voice_viz_callbacks import SpeechTokenizerVizCallback
from larry.voice.utils import collapse_ctc_emissions

WHISPER_NORMALIZER_REPO = "openai/whisper-small"
WHISPER_NORMALIZER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"


@dataclasses.dataclass
class ErrorCounts:
    """Helper dataclass for containing the results of an evaluation from the configured whisper model."""

    word_errors: int
    words: int
    char_errors: int
    chars: int


def load_index(config: SpeechSourceConfig) -> HFDataset:
    """Helper for loading an index dataset from either local parquets or huggingface."""
    if Path(config.index_path).expanduser().is_dir():
        return load_dataset(
            "parquet",
            data_files=index_split_files(config.index_path, config.split),
            split="train"
        )

    return load_dataset(config.index_path, split=config.split)


class VoiceTrainer[M: VoiceModel, C: VoiceTrainerConfig = VoiceTrainerConfig](LarryTrainer[M, C], ABC):
    """Superclass for voice trainers."""

    ...


def _normalize_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """Whisper normalization: lowercase, fillers dropped, numbers as digits, American spellings, no punctuation."""
    return whisper_normalizer(text)


@functools.cache
def _normalize_reference_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """``normalize_for_scoring`` for references, which repeat at every eval."""
    return _normalize_for_scoring(whisper_normalizer, text)


class SpeechTokenizerTrainer(VoiceTrainer[SpeechTokenizerModel, SpeechTokenizerTrainerConfig]):
    """Trainer for ``SpeechTokenizerModel`` that trains for CTC ASR."""

    model_config: SpeechTokenizerModelConfig

    @functools.cached_property
    def tokenizer(self) -> TextTokenizer:
        """Loads the SentencePieceTokenizer trained on this trainer's weighted training dataset mixture."""
        return SentencePieceTokenizer(self.config.tokenizer_model_path)

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Wraps _forward_pass to first keep a running count of total tokens seen by the model in training."""
        self.count_tokens(batch["n_target_tokens"])
        return self._forward_pass(batch)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Does the model and criterion forward and accrues statistics before returning a dataclass."""
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
        """Helper property for obtaining the total vocab size from the FSQ model config."""
        return self.model_config.fsq_levels ** self.model_config.fsq_dims

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        """On step end, add a loss_by_tokens in addition to the rest of the training callbacks."""
        if step % self.config.logging_steps == 0:
            self.metrics.add_scalar("train/loss_by_tokens", train_loss.item(), self.sync_tokens_seen())

        return self._callback_func(lambda c: c.on_step_end, train_loss, step)

    def add_training_callbacks(self) -> None:
        """Adds the default training callbacks as well as an infrequent callback for training speaker leakage probes."""
        super().add_training_callbacks()
        probes_config = self.config.speaker_leakage_probes_config
        if probes_config is not None:
            self.training_callbacks.append(voice_callbacks.SpeakerLeakageProbesCallback(probes_config))

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        """Whisper English scoring normalizer, with the British->American spelling map shipped beside its tokenizer."""
        return SpeechTokenizerVizCallback(viz_steps)

    @functools.cached_property
    def whisper_normalizer(self) -> EnglishTextNormalizer:
        """Downloads, initializes, and caches an instance of the EnglishTextNormalizer used by whisper."""
        path = hf_hub_download(WHISPER_NORMALIZER_REPO, "normalizer.json", revision=WHISPER_NORMALIZER_REVISION)
        with open(path, encoding="utf-8") as f:
            return EnglishTextNormalizer(json.load(f))

    def make_criterion(self) -> Criterion:
        """Initializes the CTC criterion."""
        if self.tokenizer.vocab_size() != self.model_config.text_vocab_size:
            raise ValueError(
                f"vocab_sizes do not match: "
                f"tokenizer={self.tokenizer.vocab_size()}, model_config={self.model_config.text_vocab_size}"
            )

        return CTCCriterion(CTCCriterionConfig(blank=self.model_config.text_vocab_size), self.model)

    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Performs the eval forward while collecting additional accuracy metrics."""
        output = self.model(batch["waveforms"], batch["waveform_lengths"])
        criterion = cast(CTCCriterion, self.criterion)
        sample_losses = criterion.sample_losses(output, batch["targets"], batch["target_lengths"])
        loss = (sample_losses / batch["target_lengths"].clamp(min=1)).mean()
        return StepOutput(loss=loss, metrics=output.metrics, sample_losses=sample_losses, model_output=output)

    def eval_group_names(self) -> list[str]:
        """Returns the name of all eval dataset sources for eval metrics tracking."""
        return [
            source.name
            for source in self.config.mixture_config.eval_sources
        ]

    def eval_group_totals(
            self, batch: Mapping[str, Any], output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Decomposes an eval batch into source ids, sample losses, and target lengths."""
        if output.sample_losses is None:
            return None

        return batch["source_ids"], output.sample_losses, batch["target_lengths"]

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        """Initializes tensors for storing eval statistics in; updated in update_eval_statistics after eval."""
        n_sources = len(self.config.mixture_config.eval_sources)
        counts = torch.zeros((n_sources, self.codebook_size), dtype=torch.long, device=self.accelerator.device)
        # per source: [raw, normalized] x [word errors, reference words, char errors, reference chars]
        errors = torch.zeros((n_sources, 2, 4), dtype=torch.float64, device=self.accelerator.device)
        return {"fsq_counts": counts, "error_counts": errors}

    def update_eval_statistics(
            self, statistics: dict[str, torch.Tensor], batch: Mapping[str, Any], output: StepOutput
    ) -> None:
        """Updates the eval statistic storage tensors provided for dataset source metrics."""
        model_output = cast(LarrySpeechTokenizerModelOutput, output.model_output)
        statistics["fsq_counts"] += code_counts(
            model_output.token_ids,
            model_output.frame_lengths,
            self.codebook_size,
            batch["source_ids"],
            len(self.config.mixture_config.eval_sources),
        ).to(statistics["fsq_counts"].device)

        blank = self.model_config.text_vocab_size
        greedy = model_output.ctc_log_probs.argmax(dim=-1).cpu()
        frame_lengths = model_output.frame_lengths.cpu().tolist()
        targets = batch["targets"].cpu()
        target_lengths = batch["target_lengths"].cpu().tolist()
        rows = []
        for i, (n_frames, n_targets) in enumerate(zip(frame_lengths, target_lengths, strict=False)):
            hypothesis = self.tokenizer.decode(collapse_ctc_emissions(greedy[i, :n_frames], blank))
            reference = self.tokenizer.decode(targets[i, :n_targets].tolist())
            raw = self._error_counts(hypothesis, reference)
            normalized = self._error_counts(
                _normalize_for_scoring(self.whisper_normalizer, hypothesis),
                _normalize_reference_for_scoring(self.whisper_normalizer, reference),
            )
            rows.append([dataclasses.astuple(raw), dataclasses.astuple(normalized)])

        errors = statistics["error_counts"]
        batch_errors = torch.tensor(rows, dtype=errors.dtype, device=errors.device)
        errors.index_add_(0, batch["source_ids"].to(errors.device), batch_errors)

    @staticmethod
    def _error_counts(hypothesis: str, reference: str) -> ErrorCounts:
        """Word and character edit distances of ``hypothesis`` against ``reference``, with the reference lengths."""
        reference_words = reference.split()
        reference_chars = " ".join(reference_words)
        return ErrorCounts(
            word_errors=Levenshtein.distance(hypothesis.split(), reference_words),
            words=len(reference_words),
            char_errors=Levenshtein.distance(" ".join(hypothesis.split()), reference_chars),
            chars=len(reference_chars),
        )

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        """Finalizes eval metrics by producing a ``MetricDict``."""
        metrics = MetricDict()
        counts = statistics["fsq_counts"]
        dims, levels = self.model_config.fsq_dims, self.model_config.fsq_levels
        for name, value in codebook_statistics(counts.sum(dim=0), dims, levels).items():
            metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}", scalar=value))

        for source, source_counts in zip(self.config.mixture_config.eval_sources, counts, strict=False):
            for name, value in codebook_statistics(source_counts, dims, levels).items():
                if not name.startswith("dim"):
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}/{source.name}", scalar=value))

        errors = statistics["error_counts"]
        groups = [("", errors.sum(dim=0))] + [
            (f"/{source.name}", source_errors)
            for source, source_errors in zip(self.config.mixture_config.eval_sources, errors, strict=False)
        ]

        for suffix, group in groups:
            for variant, (wers, words, cers, chars) in zip(("", "_normalized"), group.tolist(), strict=False):
                if words > 0:
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/wer{variant}{suffix}", scalar=wers / words))

                if chars > 0:
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/cer{variant}{suffix}", scalar=cers / chars))

        return metrics

    def make_dataset_dict(self) -> Mapping[str, Dataset]:
        """Loads the indexed dataset sources from config for this ``SpeechTokenizerTrainer``."""
        if self.model_config.sample_rate != self.config.mixture_config.sample_rate:
            raise ValueError(
                f"sample_rates do not match: model_config={self.model_config.sample_rate}, "
                f"dataset_config={self.config.mixture_config.sample_rate}"
            )

        if len(self.config.mixture_config.train_sources) == 0:
            raise ValueError("train_sources is empty")

        if len(self.config.mixture_config.eval_sources) == 0:
            raise ValueError("eval_sources is empty")

        mixture = self.config.mixture_config
        return {
            "train": self._load_indexed(mixture.train_sources, tuple(self.config.speed_augment_factors)),
            "validation": self._load_indexed(mixture.eval_sources),
        }

    def _load_indexed(
            self,
            configs: list[SpeechSourceConfig],
            speed_factors: tuple[float, ...] = (1.0,),
    ) -> IndexedSpeechDataset:
        """Helper for loading the indexed dataset source configuration. Index datasets can be materialized."""
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
                    cache_dir=config.source_cache_dir,
                )
                sources.append(source.cast_column(config.audio_column, Audio(decode=False)))
            indexes.append(index)
        return IndexedSpeechDataset(
            configs,
            indexes,
            sources,
            self.config.mixture_config.sample_rate,
            speed_factors=speed_factors,
        )

    def make_train_dataloader(self, dataset: Mapping[str, Dataset]) -> DataLoader:
        """Loads the weighted dataloader for the train dataset."""
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
        """Loads the weighted dataloader for the validation dataset."""
        validation = cast(IndexedSpeechDataset, dataset["validation"])

        _, durations = validation.source_ids_and_durations()
        return DataLoader(
            validation,
            batch_sampler=self.duration_batches(durations),
            collate_fn=self.make_collate_fn(),
            num_workers=self.config.num_dataloader_workers,
            persistent_workers=self.config.num_dataloader_workers > 0,
        )

    def duration_batches(self, durations: torch.Tensor) -> list[list[int]]:
        """Groups row positions, shortest first, into batches whose padded length fits ``max_batch_seconds``."""
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

        return batches

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """Initializes the collation function for producing formatted batches that this trainer consumes."""
        return SpeechCTCCollator(self.tokenizer)

    def update_provenance(self, provenance: Provenance) -> None:
        """Updates provenance for this model to add all datasets it was trained on in some recognizable form."""
        for source in self.config.mixture_config.train_sources:
            provenance.ancestry.datasets.add(source.index_path)
