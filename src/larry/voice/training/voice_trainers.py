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

from larry.common.config.data.data_configs import DatasetSourceConfig
from larry.common.criteria import criterion
from larry.common.criteria.criterion import Criterion
from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.data.dataloading.weighted_batch_sampler import WeightedBatchSampler, length_batches
from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.model.model import LarryModelOutput
from larry.common.training.trainers import LarryTrainer, StepOutput, VizCallback
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer, TextTokenizer
from larry.voice.config.criterion.voice_criterion_configs import CTCCriterionConfig
from larry.voice.config.data.voice_data_configs import (
    ASRSourceConfig,
    SpeechMixtureConfig,
    SpeechSourceConfig,
    TTSSourceConfig,
)
from larry.voice.config.training.voice_trainer_configs import (
    SpeechDecoderTrainerConfig,
    SpeechEncoderDecoderTrainerConfig,
    SpeechEncoderTrainerConfig,
    VocoderTrainerConfig,
    VoiceTrainerConfig,
)
from larry.voice.criteria.ctc import CTCCriterion
from larry.voice.criteria.flow_matching import SpeechDecoderCriterion, VocoderCriterion
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
from larry.voice.model.voice_modules import MelAugmenter
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

    @property
    def mixture(self) -> SpeechMixtureConfig[Any]:
        # the trainer config's mixture parameter is erased to Any by the config bound; every voice mixture is one of these
        return self.config.mixture_config

    def duration_batches(self, lengths: torch.Tensor) -> list[list[int]]:
        return length_batches(torch.argsort(lengths).tolist(), lengths, self.mixture.max_batch_seconds)


def _normalize_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """Whisper normalization: lowercase, fillers dropped, numbers as digits, American spellings, no punctuation."""
    return whisper_normalizer(text)


@functools.cache
def _normalize_reference_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """``normalize_for_scoring`` for references, which repeat at every eval."""
    return _normalize_for_scoring(whisper_normalizer, text)


class SpeechEncoderDecoderTrainerBase[
M: VoiceModel = VoiceModel,
T: SpeechEncoderDecoderTrainerConfig[Any] = SpeechEncoderDecoderTrainerConfig[Any],
S: SpeechSourceConfig = SpeechSourceConfig,
D: IndexedSpeechDataset[Any] = IndexedSpeechDataset[Any],
O: LarryModelOutput = LarryModelOutput,
](VoiceTrainer[M, T, S, D], ABC):
    """Trainer for ``SpeechTokenizerModel`` that trains for CTC ASR."""

    def make_dataset_dict(self) -> Mapping[str, D]:
        """Loads the indexed dataset sources from config for this ``SpeechTokenizerTrainer``."""
        if self.unwrapped_model.config.sample_rate != self.mixture.sample_rate:
            raise ValueError(
                f"sample_rates do not match: model_config={self.unwrapped_model.config.sample_rate}, "
                f"mixture_config={self.mixture.sample_rate}"
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
            max_batch_seconds=self.mixture.max_batch_seconds,
            pool_size=self.config.mixture_config.pool_size,
            seed=self.config.mixture_config.seed,
        )

    def make_eval_sampler(self, dataset: D) -> Sampler | Iterable:
        """Groups row positions, shortest first, into batches whose padded length fits ``max_batch_seconds``."""
        _, durations = dataset.source_ids_and_durations()
        return self.duration_batches(durations)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        """Shared by train and eval: the criterion's terms give the loss and every logged component of it."""
        output = self._model_forward(batch)
        step_output = self.step_output(output, self.criterion.loss_terms(output, **self.criterion_kwargs(batch)))
        if self.model.training:
            step_output.metrics.update(self.train_metrics(output))

        return step_output

    def train_metrics(self, output: O) -> MetricDict:
        """Extra metrics only worth computing on training batches."""
        return MetricDict()

    @abstractmethod
    def _model_forward(self, batch: Mapping[str, Any]) -> O:
        ...

    @abstractmethod
    def criterion_kwargs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        ...


class SpeechEncoderTrainer(
    SpeechEncoderDecoderTrainerBase[
        SpeechEncoderModel,
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
            self.mixture.sample_rate,
            speed_factors=(1.0,) if is_eval else tuple(self.config.speed_augment_factors),
            seed=self.config.seed,
            process_index=PartialState().process_index,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """Initializes the collation function for producing formatted batches that this trainer consumes."""
        return SpeechCTCCollator(self.tokenizer)

    def make_criterion(self) -> Criterion:
        """Initializes the CTC criterion."""
        if self.tokenizer.vocab_size() != self.unwrapped_model.config.vocab_size:
            raise ValueError(
                f"vocab_sizes do not match: "
                f"tokenizer={self.tokenizer.vocab_size()}, model_config={self.unwrapped_model.config.vocab_size}"
            )

        return CTCCriterion(CTCCriterionConfig(
            blank=self.unwrapped_model.config.vocab_size,
            ctc_layer_loss_weights=self.config.ctc_layer_loss_weights,
        ))

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        """Whisper English scoring normalizer, with the British->American spelling map shipped beside its tokenizer."""
        return voice_viz_callbacks.SpeechEncoderVizCallback(
            self,
            viz_steps,
            blank=self.unwrapped_model.config.vocab_size,
            tokenizer=self.tokenizer
        )

    def train_metrics(self, output: SpeechEncoderModelOutput) -> MetricDict:
        """Codebook statistics of the training batch, every ``logging_steps``; eval gathers them over the whole set."""
        metrics = MetricDict()
        if self.train_state.global_step % self.config.logging_steps == 0:
            counts = code_counts(output.unit_ids, output.frame_lengths, self.unwrapped_model.config.n_units)[0]
            statistics = codebook_statistics(counts, self.unwrapped_model.config.fsq_levels)
            for name, value in statistics.items():
                if not name.startswith("dim"):
                    metrics.add_metric(ScalarMetricEntry(tag=f"train/fsq/{name}", scalar=value))

        return metrics

    def _model_forward(self, batch: Mapping[str, Any]) -> SpeechEncoderModelOutput:
        return self.model(batch["waveforms"], batch["waveform_lengths"])

    def criterion_kwargs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {"targets": batch["targets"], "target_lengths": batch["target_lengths"]}

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        """Initializes tensors for storing eval statistics in; updated in update_eval_statistics after eval."""
        n_sources = len(self.config.mixture_config.eval_sources)
        counts = torch.zeros(
            (n_sources, self.unwrapped_model.config.n_units),
            dtype=torch.long,
            device=self.accelerator.device
        )

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
            self.unwrapped_model.config.n_units,
            batch["source_ids"],
            len(self.config.mixture_config.eval_sources),
        ).to(statistics["fsq_counts"].device)

        blank = self.unwrapped_model.config.vocab_size
        greedy = model_output.logits.argmax(dim=-1).cpu()
        ctc_lengths = model_output.ctc_lengths.cpu().tolist()
        targets = batch["targets"].cpu()
        target_lengths = batch["target_lengths"].cpu().tolist()
        rows = []
        for i, (n_frames, n_targets) in enumerate(zip(ctc_lengths, target_lengths, strict=True)):
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
        levels = self.unwrapped_model.config.fsq_levels
        for name, value in codebook_statistics(counts.sum(dim=0), levels).items():
            metrics.add_metric(ScalarMetricEntry(tag=f"eval/fsq/{name}", scalar=value))

        for source, source_counts in zip(self.config.mixture_config.eval_sources, counts, strict=True):
            for name, value in codebook_statistics(source_counts, levels).items():
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
            self.mixture.sample_rate,
            self.speaker_mapping,
        )

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        """Initializes the collation function for producing formatted batches that this trainer consumes."""
        return SpeechDecoderCollator()

    def make_criterion(self) -> criterion.Criterion:
        return SpeechDecoderCriterion(self.config.criterion)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return voice_viz_callbacks.SpeechDecoderVizCallback(self, viz_steps)

    def _model_forward(self, batch: Mapping[str, Any]) -> SpeechDecoderModelOutput:
        return self.model(
            batch["unit_ids"],
            batch["unit_id_lengths"],
            batch["speaker_ids"],
            batch["waveforms"],
            batch["waveform_lengths"],
        )

    def criterion_kwargs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        # the KL weight warms up with the step, so eval weighs it as training currently does
        return {"step": self.train_state.global_step}

    def count_tokens(self, batch: Mapping[str, Any]) -> None:
        self._pending_tokens += int(batch["unit_id_lengths"].sum()) * self.unwrapped_model.config.unit_upsample


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
            self.mixture.sample_rate,
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
            crop_samples=round(self.config.crop_seconds * self.mixture.sample_rate),
            crops_per_clip=self.config.crops_per_clip,
            seed=self.config.seed,
            process_index=PartialState().process_index,
        )

    def make_eval_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return WaveformCollator()

    @functools.cached_property
    def mel_augmenter(self) -> MelAugmenter:
        return MelAugmenter(self.config.mel_augment)

    def make_criterion(self) -> criterion.Criterion:
        return VocoderCriterion(self.config.criterion, self.unwrapped_model.config).to(self.accelerator.device)

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        waveforms = batch["waveforms"].to(self.accelerator.device)
        waveform_lengths = batch["waveform_lengths"].to(self.accelerator.device)

        model = self.unwrapped_model
        mels, mel_lengths = model.frontend(waveforms, waveform_lengths)
        spec = model.codec.encode(waveforms)
        conditioning = self.mel_augmenter(mels) if self.model.training else mels
        output = self.model(spec, conditioning, mel_lengths)
        terms = self.criterion.loss_terms(
            output,
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            mels=mels,
            mel_lengths=mel_lengths,
        )
        return self.step_output(output, terms)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        viz = self.config.viz
        return voice_viz_callbacks.VocoderVizCallback(
            self,
            viz_steps,
            examples_per_source=viz.examples_per_source,
            generation_steps=viz.generation_steps,
            step_sweep=tuple(viz.step_sweep),
            sway_coefficient=viz.sway_coefficient,
            seed=viz.seed,
            world=viz.world,
        )
