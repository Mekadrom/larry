import dataclasses
from collections.abc import Iterator
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from larry.text.tokenizer.text_tokenizer import TextTokenizer


@dataclasses.dataclass(kw_only=True)
class SpeechSourceConfig:
    name: str
    index_path: str  # huggingface repo or local dir: rows with id, duration_s, text_original, speaker_id, etc.
    source_path: str | None = None
    source_name: str | None = None
    source_revision: str | None = None
    split: str = "train"
    source_split: str = "train"
    audio_column: str = "audio"
    duration_column: str = "duration_s"
    text_column: str = "text_normalized"
    weight: float = 1.0  # target share of audio time, not of rows


@dataclasses.dataclass(kw_only=True)
class SpeechMixtureConfig:
    train_sources: list[SpeechSourceConfig] = dataclasses.field(default_factory=list)
    eval_sources: list[SpeechSourceConfig] = dataclasses.field(default_factory=list)
    sample_rate: int = 16000
    max_batch_seconds: float = 400.0  # padded seconds per batch
    pool_size: int = 20000  # clips drawn per sampler epoch
    seed: int = 42


class WeightedBucketBatchSampler(Sampler[list[int]]):
    def __init__(
            self,
            source_ids: torch.Tensor,
            durations: torch.Tensor,
            weights: list[float],
            max_batch_seconds: float,
            pool_size: int,
            seed: int,
    ) -> None:
        self.durations = durations.double()
        self.max_batch_seconds = max_batch_seconds
        self.pool_size = pool_size
        self.seed = seed
        self.epoch = 0

        self.rows_by_source = [
            torch.nonzero(source_ids == s).flatten()
            for s in range(len(weights))
        ]

        # time shares -> per-draw probabilities: short-clip sources need more draws per hour
        per_draw = []
        for s, weight in enumerate(weights):
            mean_duration = float(self.durations[self.rows_by_source[s]].mean())
            per_draw.append(weight / mean_duration)
        probs = torch.tensor(per_draw, dtype=torch.float64)
        self.source_probs = probs / probs.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(self.source_probs, self.pool_size, replacement=True, generator=generator)

        pool_parts = []
        for s, rows in enumerate(self.rows_by_source):
            count = int((draws == s).sum())
            if count == 0:
                continue
            picks = torch.randint(len(rows), (count,), generator=generator)
            pool_parts.append(rows[picks])

        pool = torch.cat(pool_parts)
        pool = pool[torch.argsort(self.durations[pool])]

        batches: list[list[int]] = []
        batch: list[int] = []
        for row in pool.tolist():
            # ascending durations: this row is the longest so far, so this is the padded size
            padded_seconds = (len(batch) + 1) * float(self.durations[row])
            if len(batch) > 0 and padded_seconds > self.max_batch_seconds:
                batches.append(batch)
                batch = []
            batch.append(row)
        if len(batch) > 0:
            batches.append(batch)

        order = torch.randperm(len(batches), generator=generator).tolist()
        return [
            batches[i]
            for i in order
        ]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


class SpeechCTCCollator:
    def __init__(self, text_tokenizer: TextTokenizer) -> None:
        self.text_tokenizer = text_tokenizer

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        waveforms = [
            item["waveform"]
            for item in items
        ]
        targets = [
            torch.tensor(self.text_tokenizer.encode(item["text"]), dtype=torch.long)
            for item in items
        ]
        return {
            "waveforms": pad_sequence(waveforms, batch_first=True),
            "waveform_lengths": torch.tensor([len(w) for w in waveforms]),
            "targets": pad_sequence(targets, batch_first=True),
            "target_lengths": torch.tensor([len(t) for t in targets]),
        }
