from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from larry.common.data.dataloading.weighted_batch_sampler import (
    LengthBucketingWeightedBatchSampler,
    WeightedBatchSampler,
)
from larry.common.data.dataloading.worker_random import WorkerRandom
from larry.text.tokenizer.text_tokenizer import TextTokenizer


class AudioDurationBucketingWeightedBatchSampler(LengthBucketingWeightedBatchSampler):
    """Implementation of ``ResumableBatchSampler`` that allows configuration of weights per dataset source."""

    def __init__(
            self,
            source_ids: torch.Tensor,
            durations: torch.Tensor,
            weights: list[float],
            max_batch_seconds: float,
            pool_size: int,
            seed: int,
    ) -> None:
        """Parses the configuration into weights per dataset source and initializes the sampler."""
        super().__init__(source_ids, durations, weights, max_batch_seconds, pool_size, seed)
        self.durations = durations
        self.max_batch_seconds = max_batch_seconds


class ClipCropWeightedBatchSampler(WeightedBatchSampler):
    def __init__(
            self,
            source_ids: torch.Tensor,
            durations: torch.Tensor,
            weights: list[float],
            clips_per_batch: int,
            crop_seconds: float,
            crops_per_clip: int,
            pool_size: int,
            seed: int,
    ) -> None:
        self.durations = durations
        self.clips_per_batch = clips_per_batch
        self.crop_seconds = crop_seconds
        self.crops_per_clip = crops_per_clip
        super().__init__(source_ids, weights, pool_size, seed)

    def cropped_seconds(self, durations: torch.Tensor) -> torch.Tensor:
        crops = torch.div(durations, self.crop_seconds, rounding_mode="floor").clamp(min=1, max=self.crops_per_clip)
        return torch.where(durations >= self.crop_seconds, crops * self.crop_seconds, durations)

    def recalculate_source_probs(self) -> torch.Tensor:
        per_draw = []
        for s, weight in enumerate(self.weights):
            per_draw.append(weight / float(self.cropped_seconds(self.durations[self.rows_by_source[s]]).mean()))
        return torch.tensor(per_draw, dtype=torch.float64)

    def sort_pool(self, pool: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return pool[torch.randperm(len(pool), generator=generator)]

    def make_batches(self, pool: torch.Tensor) -> list[list[int]]:
        return [batch.tolist() for batch in pool.split(self.clips_per_batch)]


class RandomCropCollator:
    def __init__(self, crop_samples: int, crops_per_clip: int, seed: int, process_index: int = 0) -> None:
        self.crop_samples = crop_samples
        self.crops_per_clip = crops_per_clip
        self.seed = seed
        self.process_index = process_index
        self.random = WorkerRandom(seed, process_index)

    def crop_starts(self, n_samples: int) -> list[int]:
        if n_samples <= self.crop_samples:
            return [0]

        n_crops = min(self.crops_per_clip, n_samples // self.crop_samples)
        segment = n_samples // n_crops
        return [i * segment + self.random.get().randint(0, segment - self.crop_samples) for i in range(n_crops)]

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = []
        waveform_lengths = []
        source_ids = []
        for item in items:
            waveform = item["waveform"]
            for start in self.crop_starts(len(waveform)):
                crop = waveform[start:start + self.crop_samples]
                waveforms.append(F.pad(crop, (0, self.crop_samples - len(crop))))
                waveform_lengths.append(len(crop))
                source_ids.append(item["source"])

        return {
            "waveforms": torch.stack(waveforms),
            "waveform_lengths": torch.tensor(waveform_lengths),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
        }


class WaveformCollator:
    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [item["waveform"] for item in items]
        return {
            "waveforms": pad_sequence(waveforms, batch_first=True),
            "waveform_lengths": torch.tensor([len(w) for w in waveforms]),
            "source_ids": torch.tensor([item["source"] for item in items], dtype=torch.long),
        }


class SpeechCTCCollator:
    """Massages sampled examples into stacked batches for training a speech tokenizer with CTC targets."""

    def __init__(self, text_tokenizer: TextTokenizer) -> None:
        """Takes a text tokenizer for the target text."""
        self.text_tokenizer = text_tokenizer

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalizes and converts lists of numbers to tensors where applicable. The true batch is returned."""
        waveforms = []
        waveform_lengths = []
        targets = []
        target_lengths = []
        source_ids = []
        for item in items:
            waveform = item["waveform"]
            waveforms.append(waveform)
            waveform_lengths.append(len(waveform))

            target = self.text_tokenizer.encode(item["text"])
            targets.append(torch.tensor(target, dtype=torch.long))
            target_lengths.append(len(target))

            source_ids.append(item["source"])

        return {
            "waveforms": pad_sequence(waveforms, batch_first=True),
            "waveform_lengths": torch.tensor(waveform_lengths),
            "targets": pad_sequence(targets, batch_first=True),
            "target_lengths": torch.tensor(target_lengths),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "n_target_tokens": sum(target_lengths),
        }


class SpeechDecoderCollator:
    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = []
        waveform_lengths = []
        unit_ids = []
        unit_id_lengths = []
        source_ids = []
        speaker_ids = []
        for item in items:
            waveform = item["waveform"]
            waveforms.append(waveform)
            waveform_lengths.append(len(waveform))

            units = item["unit_ids"]
            unit_ids.append(torch.tensor(units, dtype=torch.long))
            unit_id_lengths.append(len(units))

            source_ids.append(item["source"])
            speaker_ids.append(item["speaker_id"])

        return {
            "waveforms": pad_sequence(waveforms, batch_first=True),
            "waveform_lengths": torch.tensor(waveform_lengths),
            "unit_ids": pad_sequence(unit_ids, batch_first=True),
            "unit_id_lengths": torch.tensor(unit_id_lengths),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
        }
