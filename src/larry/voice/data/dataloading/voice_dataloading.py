import dataclasses
import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from larry.common.data.samplers import ResumableBatchSampler
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
    pool_size: int = 20000  # clips drawn per sampler chunk
    seed: int = 42


class WeightedBucketBatchSampler(ResumableBatchSampler):
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

        self.rows_by_source = [
            torch.nonzero(source_ids == s).flatten()
            for s in range(len(weights))
        ]
        for s, rows in enumerate(self.rows_by_source):
            if len(rows) == 0:
                raise ValueError(f"source {s} has no rows")

        # time shares -> per-draw probabilities: short-clip sources need more draws per hour
        per_draw = []
        for s, weight in enumerate(weights):
            mean_duration = float(self.durations[self.rows_by_source[s]].mean())
            per_draw.append(weight / mean_duration)
        probs = torch.tensor(per_draw, dtype=torch.float64)
        self.source_probs = probs / probs.sum()

        self._counts_by_chunk: list[torch.Tensor] = []
        self._batches_per_chunk: list[int] = []
        self._permutations: dict[int, tuple[int, torch.Tensor]] = {}
        self.start_chunk = 0
        self.start_offset = 0

    def start_at(self, consumed_batches: int) -> None:
        chunk = 0
        while consumed_batches >= self._batch_count(chunk):
            consumed_batches -= self._batch_count(chunk)
            chunk += 1
        self.start_chunk = chunk
        self.start_offset = consumed_batches

    def _batch_count(self, chunk: int) -> int:
        while len(self._batches_per_chunk) <= chunk:
            self._batches_per_chunk.append(len(self._batches(len(self._batches_per_chunk))))
        return self._batches_per_chunk[chunk]

    def _source_counts(self, chunk: int) -> tuple[torch.Tensor, torch.Generator]:
        """Rows each source contributes to ``chunk``, and that chunk's generator for shuffling batch order."""
        generator = torch.Generator().manual_seed(self.seed + chunk)
        draws = torch.multinomial(self.source_probs, self.pool_size, replacement=True, generator=generator)
        return torch.bincount(draws, minlength=len(self.rows_by_source)), generator

    def _start_positions(self, chunk: int) -> torch.Tensor:
        """Each source's cursor at the start of ``chunk``: rows it contributed in all earlier chunks."""
        while len(self._counts_by_chunk) < chunk:
            counts, _ = self._source_counts(len(self._counts_by_chunk))
            self._counts_by_chunk.append(counts)
        starts = torch.zeros(len(self.rows_by_source), dtype=torch.long)
        for counts in self._counts_by_chunk[:chunk]:
            starts += counts
        return starts

    def _permutation(self, s: int, cycle: int) -> torch.Tensor:
        # keyed by (seed, source, cycle) and not by chunk: a cycle read across two chunks must be the same shuffle
        cached = self._permutations.get(s)
        if cached is not None and cached[0] == cycle:
            return cached[1]
        seed = int(np.random.SeedSequence([self.seed, s, cycle]).generate_state(1)[0])
        generator = torch.Generator().manual_seed(seed)
        rows = self.rows_by_source[s]
        permutation = rows[torch.randperm(len(rows), generator=generator)]
        self._permutations[s] = (cycle, permutation)
        return permutation

    def _take(self, s: int, start: int, count: int) -> torch.Tensor:
        """Rows ``start`` to ``start + count`` of source ``s``'s stream of back-to-back shuffled passes."""
        n = len(self.rows_by_source[s])
        parts = []
        position = start
        end = start + count
        while position < end:
            cycle, offset = divmod(position, n)
            take = min(n - offset, end - position)
            parts.append(self._permutation(s, cycle)[offset:offset + take])
            position += take
        if not parts:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(parts)

    def _batches(self, chunk: int) -> list[list[int]]:
        counts, generator = self._source_counts(chunk)
        starts = self._start_positions(chunk)

        pool = torch.cat([
            self._take(s, int(starts[s]), int(counts[s]))
            for s in range(len(self.rows_by_source))
        ])
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
        for chunk in itertools.count(self.start_chunk):
            batches = self._batches(chunk)
            yield from batches[self.start_offset if chunk == self.start_chunk else 0:]


class SpeechCTCCollator:
    def __init__(self, text_tokenizer: TextTokenizer) -> None:
        self.text_tokenizer = text_tokenizer

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
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
            "source_ids": torch.tensor([item["source"] for item in items], dtype=torch.long),
            # a plain int stays on the cpu when the batch is moved to the device, so reading it never syncs
            "n_target_tokens": sum(len(t) for t in targets),
        }
