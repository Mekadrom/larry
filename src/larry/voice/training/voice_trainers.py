import dataclasses
import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import PartialState
from datasets import Audio
from datasets import Dataset as HFDataset
from huggingface_hub import hf_hub_download
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Sampler
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.config.data.data_configs import DatasetSourceConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.criteria import criterion
from larry.common.criteria.criterion import Criterion, SamplableCriterion
from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.data.dataloading.weighted_batch_sampler import WeightedBatchSampler
from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.model.model import LarryModelOutput
from larry.common.training.trainers import LarryTrainer, StepOutput, VizCallback
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer, TextTokenizer
from larry.voice.config.criterion.voice_criterion_configs import CTCCriterionConfig
from larry.voice.config.data.voice_data_configs import ASRSourceConfig, SpeechSourceConfig, TTSSourceConfig
from larry.voice.config.model.voice_model_configs import (
    SpeechDecoderModelConfig,
    SpeechEncoderDecoderModelConfig,
    SpeechEncoderModelConfig,
)
from larry.voice.config.training.voice_trainer_configs import (
    SpeechDecoderTrainerConfig,
    SpeechEncoderDecoderTrainerConfig,
    SpeechEncoderTrainerConfig,
    VocoderTrainerConfig,
    VoiceTrainerConfig,
)
from larry.voice.criteria.ctc import CTCCriterion
from larry.voice.criteria.flow_matching import SpeechDecoderCriterion
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedASRDataset, IndexedSpeechDataset, IndexedTTSDataset
from larry.voice.data.dataloading.voice_dataloading import (
    AudioDurationBucketingWeightedBatchSampler,
    ClipCropWeightedBatchSampler,
    RandomCropCollator,
    SpeechCTCCollator,
    SpeechDecoderCollator,
    WaveformCollator,
)
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from larry.voice.model.fsq import code_counts, codebook_statistics
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.speech_encoder import SpeechEncoderModel
from larry.voice.model.vocoder import LarryVocoder
from larry.voice.model.voice_models import SpeechDecoderModelOutput, SpeechEncoderModelOutput, VoiceModel
from larry.voice.training import voice_callbacks, voice_viz_callbacks
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


def error_counts(hypothesis: str, reference: str) -> ErrorCounts:
    """Word and character edit distances of ``hypothesis`` against ``reference``, with the reference lengths."""
    reference_words = reference.split()
    reference_chars = " ".join(reference_words)
    return ErrorCounts(
        word_errors=Levenshtein.distance(hypothesis.split(), reference_words),
        words=len(reference_words),
        char_errors=Levenshtein.distance(" ".join(hypothesis.split()), reference_chars),
        chars=len(reference_chars),
    )


class VoiceTrainer[
M: VoiceModel,
C: VoiceTrainerConfig[Any] = VoiceTrainerConfig[Any],
S: DatasetSourceConfig = DatasetSourceConfig,
D: IndexedDataset[Any] = IndexedDataset[Any],
](LarryTrainer[M, C, S, D], ABC):
    """Superclass for voice trainers."""

    def duration_batches(self, durations: torch.Tensor) -> list[list[int]]:
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


def _normalize_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """Whisper normalization: lowercase, fillers dropped, numbers as digits, American spellings, no punctuation."""
    return whisper_normalizer(text)


@functools.cache
def _normalize_reference_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """``normalize_for_scoring`` for references, which repeat at every eval."""
    return _normalize_for_scoring(whisper_normalizer, text)


class SpeechEncoderDecoderTrainerBase[
M: VoiceModel = VoiceModel,
C: SpeechEncoderDecoderModelConfig = SpeechEncoderDecoderModelConfig,
T: SpeechEncoderDecoderTrainerConfig = SpeechEncoderDecoderTrainerConfig,
S: SpeechSourceConfig = SpeechSourceConfig,
D: IndexedSpeechDataset[Any] = IndexedSpeechDataset[Any],
O: LarryModelOutput = LarryModelOutput,
](VoiceTrainer[M, T, S, D], ABC):
    """Trainer for ``SpeechTokenizerModel`` that trains for CTC ASR."""

    model_config: C

    def make_dataset_dict(self) -> Mapping[str, D]:
        """Loads the indexed dataset sources from config for this ``SpeechTokenizerTrainer``."""
        if self.model_config.sample_rate != self.config.mixture_config.sample_rate:
            raise ValueError(
                f"sample_rates do not match: model_config={self.model_config.sample_rate}, "
                f"mixture_config={self.config.mixture_config.sample_rate}"
            )
        return super().make_dataset_dict()

    def transform_as_source(self, config: S, dataset: HFDataset) -> HFDataset:
        return dataset.cast_column(config.audio_column, Audio(decode=False))

    def make_train_sampler(self, dataset: D, weights: list[float]) -> WeightedBatchSampler:
        source_ids, durations = dataset.source_ids_and_durations()
        return AudioDurationBucketingWeightedBatchSampler(
            source_ids,
            durations,
            weights,
            max_batch_seconds=self.config.mixture_config.max_batch_seconds,
            pool_size=self.config.mixture_config.pool_size,
            seed=self.config.mixture_config.seed,
        )

    def make_eval_sampler(self, dataset: D) -> Sampler | Iterable:
        """Groups row positions, shortest first, into batches whose padded length fits ``max_batch_seconds``."""
        _, durations = dataset.source_ids_and_durations()
        return self.duration_batches(durations)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self._model_forward(batch)
        loss = self._criterion_forward(output, batch)
        metrics = self._forward_metrics(output)
        return StepOutput(loss=loss, metrics=metrics, model_output=output)

    def eval_group_names(self) -> list[str]:
        """Returns the name of all eval dataset sources for eval metrics tracking."""
        return [
            source.name
            for source in self.config.mixture_config.eval_sources
        ]

    def update_provenance(self, provenance: Provenance) -> None:
        """Updates provenance for this model to add all datasets it was trained on in some recognizable form."""
        for source in self.config.mixture_config.train_sources:
            provenance.ancestry.datasets.add(source.provenance_name or source.index_path)

    @abstractmethod
    def _forward_metrics(self, output: O) -> MetricDict:
        ...

    @abstractmethod
    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        ...

    @abstractmethod
    def _model_forward(self, batch: Mapping[str, Any]) -> O:
        ...

    @abstractmethod
    def _criterion_forward(self, output: O, batch: Mapping[str, Any]) -> torch.Tensor:
        ...

    @abstractmethod
    def eval_group_totals(
            self, batch: Mapping[str, Any], output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        ...

    @abstractmethod
    def eval_statistics(self) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def update_eval_statistics(
            self, statistics: dict[str, torch.Tensor], batch: Mapping[str, Any], output: StepOutput
    ) -> None:
        ...

    @abstractmethod
    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        ...


class SpeechEncoderTrainer(
    SpeechEncoderDecoderTrainerBase[
        SpeechEncoderModel,
        SpeechEncoderModelConfig,
        SpeechEncoderTrainerConfig,
        ASRSourceConfig,
        IndexedASRDataset,
        SpeechEncoderModelOutput,
    ]
):
    @functools.cached_property
    def tokenizer(self) -> TextTokenizer:
        """Loads the SentencePieceTokenizer trained on this trainer's weighted training dataset mixture."""
        return SentencePieceTokenizer(self.config.tokenizer_model_path)

    @functools.cached_property
    def whisper_normalizer(self) -> EnglishTextNormalizer:
        """Downloads, initializes, and caches an instance of the EnglishTextNormalizer used by whisper."""
        path = hf_hub_download(WHISPER_NORMALIZER_REPO, "normalizer.json", revision=WHISPER_NORMALIZER_REVISION)
        with open(path, encoding="utf-8") as f:
            return EnglishTextNormalizer(json.load(f))

    def add_training_callbacks(self) -> None:
        """Adds the default training callbacks as well as an infrequent callback for training speaker leakage probes."""
        super().add_training_callbacks()
        probes_config = self.config.speaker_leakage_probes
        if probes_config is not None:
            self.training_callbacks.append(voice_callbacks.SpeakerLeakageProbesCallback(self, probes_config))

    def load_indexed(
            self,
            configs: list[ASRSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            is_eval: bool = False
    ) -> IndexedASRDataset:
        return IndexedASRDataset(
            configs,
            indexes,
            sources,
            self.config.mixture_config.sample_rate,
            speed_factors=(1.0,) if is_eval else tuple(self.config.speed_augment_factors),
            seed=self.config.seed,
            process_index=PartialState().process_index,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """Initializes the collation function for producing formatted batches that this trainer consumes."""
        return SpeechCTCCollator(self.tokenizer)

    def make_criterion(self) -> Criterion:
        """Initializes the CTC criterion."""
        if self.tokenizer.vocab_size() != self.model_config.vocab_size:
            raise ValueError(
                f"vocab_sizes do not match: "
                f"tokenizer={self.tokenizer.vocab_size()}, model_config={self.model_config.vocab_size}"
            )

        return CTCCriterion(CTCCriterionConfig(
            blank=self.model_config.vocab_size,
            ctc_layer_loss_weights=self.config.ctc_layer_loss_weights,
        ))

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        """Whisper English scoring normalizer, with the British->American spelling map shipped beside its tokenizer."""
        return voice_viz_callbacks.SpeechEncoderVizCallback(
            self,
            viz_steps,
            blank=self.model_config.vocab_size,
            tokenizer=self.tokenizer
        )

    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Performs the eval forward while collecting additional accuracy metrics."""
        output = self._model_forward(batch)
        sample_losses = cast(SamplableCriterion, self.criterion).sample_losses(
            output,
            batch["targets"],
            batch["target_lengths"]
        )
        loss = (sample_losses / batch["target_lengths"].clamp(min=1)).mean()
        return StepOutput(
            loss=loss,
            metrics=output.metrics or MetricDict({}),
            sample_losses=sample_losses,
            model_output=output
        )

    def _forward_metrics(self, output: SpeechEncoderModelOutput) -> MetricDict:
        """Does the model and criterion forward and accrues statistics before returning a dataclass."""
        metrics = MetricDict(output.metrics or {})
        if self.train_state.global_step % self.config.logging_steps == 0:
            counts = code_counts(output.unit_ids, output.frame_lengths, self.model_config.n_units)[0]
            statistics = codebook_statistics(counts, self.model_config.fsq_dims, self.model_config.fsq_levels)
            for name, value in statistics.items():
                if not name.startswith("dim"):
                    metrics.add_metric(ScalarMetricEntry(tag=f"train/fsq/{name}", scalar=value))

        return metrics

    def _model_forward(self, batch: Mapping[str, Any]) -> SpeechEncoderModelOutput:
        return self.model(batch["waveforms"], batch["waveform_lengths"])

    def _criterion_forward(self, output: SpeechEncoderModelOutput, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.criterion(output, targets=batch["targets"], target_lengths=batch["target_lengths"])

    def eval_group_totals(
            self,
            batch: Mapping[str, Any],
            output: StepOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Decomposes an eval batch into source ids, sample losses, and target lengths."""
        if output.sample_losses is None:
            return None

        return batch["source_ids"], output.sample_losses, batch["target_lengths"]

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        """Initializes tensors for storing eval statistics in; updated in update_eval_statistics after eval."""
        n_sources = len(self.config.mixture_config.eval_sources)
        counts = torch.zeros((n_sources, self.model_config.n_units), dtype=torch.long, device=self.accelerator.device)
        # per source: [raw, normalized] x [word errors, reference words, char errors, reference chars]
        errors = torch.zeros((n_sources, 2, 4), dtype=torch.float64, device=self.accelerator.device)
        return {"fsq_counts": counts, "error_counts": errors}

    def update_eval_statistics(
            self,
            statistics: dict[str, torch.Tensor],
            batch: Mapping[str, Any],
            output: StepOutput,
    ) -> None:
        """Updates the eval statistic storage tensors provided for dataset source metrics."""
        model_output = cast(SpeechEncoderModelOutput, output.model_output)
        statistics["fsq_counts"] += code_counts(
            model_output.unit_ids,
            model_output.frame_lengths,
            self.model_config.n_units,
            batch["source_ids"],
            len(self.config.mixture_config.eval_sources),
        ).to(statistics["fsq_counts"].device)

        blank = self.model_config.vocab_size
        greedy = model_output.logits.argmax(dim=-1).cpu()
        frame_lengths = model_output.frame_lengths.cpu().tolist()
        targets = batch["targets"].cpu()
        target_lengths = batch["target_lengths"].cpu().tolist()
        rows = []
        for i, (n_frames, n_targets) in enumerate(zip(frame_lengths, target_lengths, strict=True)):
            hypothesis = self.tokenizer.decode(collapse_ctc_emissions(greedy[i, :n_frames], blank))
            reference = self.tokenizer.decode(targets[i, :n_targets].tolist())
            raw = error_counts(hypothesis, reference)
            normalized = error_counts(
                _normalize_for_scoring(self.whisper_normalizer, hypothesis),
                _normalize_reference_for_scoring(self.whisper_normalizer, reference),
            )
            rows.append([dataclasses.astuple(raw), dataclasses.astuple(normalized)])

        errors = statistics["error_counts"]
        batch_errors = torch.tensor(rows, dtype=errors.dtype, device=errors.device)
        errors.index_add_(0, batch["source_ids"].to(errors.device), batch_errors)

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        """Finalizes eval metrics by producing a ``MetricDict``."""
        metrics = MetricDict()
        counts = statistics["fsq_counts"]
        dims, levels = self.model_config.fsq_dims, self.model_config.fsq_levels
        for name, value in codebook_statistics(counts.sum(dim=0), dims, levels).items():
            metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}", scalar=value))

        for source, source_counts in zip(self.config.mixture_config.eval_sources, counts, strict=True):
            for name, value in codebook_statistics(source_counts, dims, levels).items():
                if not name.startswith("dim"):
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}/{source.name}", scalar=value))

        errors = statistics["error_counts"]
        groups = [("", errors.sum(dim=0))] + [
            (f"/{source.name}", source_errors)
            for source, source_errors in zip(self.config.mixture_config.eval_sources, errors, strict=True)
        ]

        for suffix, group in groups:
            for variant, (wers, words, cers, chars) in zip(("", "_normalized"), group.tolist(), strict=True):
                if words > 0:
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/wer{variant}{suffix}", scalar=wers / words))

                if chars > 0:
                    metrics.add_metric(ScalarMetricEntry(tag=f"eval/cer{variant}{suffix}", scalar=cers / chars))

        return metrics

    def count_tokens(self, batch: Mapping[str, Any]) -> None:
        n_tokens = batch["n_target_tokens"]
        self._pending_tokens += n_tokens


class SpeechDecoderTrainer(
    SpeechEncoderDecoderTrainerBase[
        SpeechDecoderModel,
        SpeechDecoderModelConfig,
        SpeechDecoderTrainerConfig,
        TTSSourceConfig,
        IndexedTTSDataset,
        SpeechDecoderModelOutput,
    ]
):
    @functools.cached_property
    def speaker_mapping(self) -> SpeakerMapping:
        return SpeakerMapping.from_csv(Path(self.config.speaker_mapping_path))

    def load_indexed(
            self,
            configs: list[TTSSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            is_eval: bool = False
    ) -> IndexedTTSDataset:
        return IndexedTTSDataset(
            configs,
            indexes,
            sources,
            self.config.mixture_config.sample_rate,
            self.speaker_mapping,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """Initializes the collation function for producing formatted batches that this trainer consumes."""
        return SpeechDecoderCollator()

    def make_criterion(self) -> criterion.Criterion:
        return SpeechDecoderCriterion(self.config.criterion)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return voice_viz_callbacks.SpeechDecoderVizCallback(self, viz_steps)

    def eval_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self._model_forward(batch)
        decoder_criterion = cast(SpeechDecoderCriterion, self.criterion)
        return StepOutput(
            loss=decoder_criterion.flow_loss(output),
            metrics=output.metrics or MetricDict({}),
            sample_losses=decoder_criterion.sample_losses(output, output.targets),
            model_output=output,
        )

    def _forward_metrics(self, output: SpeechDecoderModelOutput) -> MetricDict:
        metrics = MetricDict(output.metrics or {})
        metrics.add_metric(ScalarMetricEntry(tag="train/kl", scalar=float(output.kl.mean())))
        prior_loss = cast(SpeechDecoderCriterion, self.criterion).prior_loss(output)
        metrics.add_metric(ScalarMetricEntry(tag="train/prior_loss", scalar=float(prior_loss)))
        return metrics

    def _model_forward(self, batch: Mapping[str, Any]) -> SpeechDecoderModelOutput:
        return self.model(
            batch["unit_ids"],
            batch["unit_id_lengths"],
            batch["speaker_ids"],
            batch["waveforms"],
            batch["waveform_lengths"],
        )

    def _criterion_forward(self, output: SpeechDecoderModelOutput, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.criterion(output, step=self.train_state.global_step)

    def eval_group_totals(
            self,
            batch: Mapping[str, Any],
            output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if output.sample_losses is None or output.model_output is None:
            return None

        elements = cast(SpeechDecoderCriterion, self.criterion).sample_elements(output.model_output)
        return batch["source_ids"], output.sample_losses, elements

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        return {}

    def update_eval_statistics(
            self,
            statistics: dict[str, torch.Tensor],
            batch: Mapping[str, Any],
            output: StepOutput
    ) -> None:
        pass

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        return MetricDict()

    def count_tokens(self, batch: Mapping[str, Any]) -> None:
        self._pending_tokens += int(batch["unit_id_lengths"].sum()) * self.model_config.unit_upsample


class VocoderTrainer(VoiceTrainer[LarryVocoder, VocoderTrainerConfig, SpeechSourceConfig, IndexedSpeechDataset]):
    def load_indexed(
            self,
            configs: list[SpeechSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            is_eval: bool = False
    ) -> IndexedSpeechDataset:
        return IndexedSpeechDataset(
            configs,
            indexes,
            sources,
            self.config.mixture_config.sample_rate,
        )

    def make_train_sampler(self, dataset: IndexedSpeechDataset, weights: list[float]) -> WeightedBatchSampler:
        source_ids, durations = dataset.source_ids_and_durations()
        return ClipCropWeightedBatchSampler(
            source_ids,
            durations,
            weights,
            clips_per_batch=self.config.clips_per_batch,
            crop_seconds=self.config.crop_seconds,
            crops_per_clip=self.config.crops_per_clip,
            pool_size=self.config.mixture_config.pool_size,
            seed=self.config.mixture_config.seed,
        )

    def make_eval_sampler(self, dataset: IndexedSpeechDataset) -> Sampler | Iterable:
        _, durations = dataset.source_ids_and_durations()
        return self.duration_batches(durations)

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return RandomCropCollator(
            crop_samples=round(self.config.crop_seconds * self.config.mixture_config.sample_rate),
            crops_per_clip=self.config.crops_per_clip,
            seed=self.config.seed,
            process_index=PartialState().process_index,
        )

    def make_eval_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return WaveformCollator()

    def make_criterion(self) -> criterion.Criterion:
        return ConditionalFlowMatchingCriterion(CriterionConfig())

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        waveforms = batch["waveforms"].to(self.accelerator.device)
        waveform_lengths = batch["waveform_lengths"].to(self.accelerator.device)

        model = self.accelerator.unwrap_model(self.model)
        mels, mel_lengths = model.frontend(waveforms, waveform_lengths)
        spec = model.codec.encode(waveforms)
        output = self.model(spec, mels, mel_lengths)
        sample_losses = cast(ConditionalFlowMatchingCriterion, self.criterion).sample_losses(output, output.targets)
        return StepOutput(
            loss=self.criterion(output), metrics=MetricDict(), sample_losses=sample_losses, model_output=output
        )

    def update_provenance(self, provenance: Provenance) -> None:
        for source in self.config.mixture_config.train_sources:
            provenance.ancestry.datasets.add(source.provenance_name or source.index_path)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return voice_viz_callbacks.VocoderVizCallback(self, viz_steps)

    def eval_group_names(self) -> list[str]:
        return [source.name for source in self.config.mixture_config.eval_sources]

    def eval_group_totals(
            self,
            batch: Mapping[str, Any],
            output: StepOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if output.sample_losses is None or output.model_output is None:
            return None

        elements = cast(ConditionalFlowMatchingCriterion, self.criterion).sample_elements(output.model_output)
        return batch["source_ids"], output.sample_losses, elements
