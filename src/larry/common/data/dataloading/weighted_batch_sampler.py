import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import torch

from larry.common.data.dataloading.samplers import ResumableBatchSampler


class WeightedBatchSampler(ResumableBatchSampler, ABC):
    def __init__(
            self,
            source_ids: torch.Tensor,
            weights: list[float],
            pool_size: int,
            seed: int,
    ) -> None:
        self.weights = weights
        self.pool_size = pool_size
        self.seed = seed

        self.rows_by_source = [
            torch.nonzero(source_ids == s).flatten()
            for s in range(len(weights))
        ]
        for s, rows in enumerate(self.rows_by_source):
            if len(rows) == 0:
                raise ValueError(f"source {s} has no rows")

        probs = self.recalculate_source_probs()
        self.source_probs = probs / probs.sum()

        self._counts_by_chunk: list[torch.Tensor] = []
        self._batches_per_chunk: list[int] = []
        self._permutations: dict[int, tuple[int, torch.Tensor]] = {}
        self.start_chunk = 0
        self.start_offset = 0

    def start_at(self, consumed_batches: int) -> None:
        """For resuming, continues from the number of consumed batches by tracking chunks."""
        chunk = 0
        while consumed_batches >= self._batch_count(chunk):
            consumed_batches -= self._batch_count(chunk)
            chunk += 1

        self.start_chunk = chunk
        self.start_offset = consumed_batches

    def _batch_count(self, chunk: int) -> int:
        """Helper for obtaining the size of a chunk in number of batches."""
        while len(self._batches_per_chunk) <= chunk:
            self._batches_per_chunk.append(len(self._batches(len(self._batches_per_chunk))))

        return self._batches_per_chunk[chunk]

    def _batches(self, chunk: int) -> list[list[int]]:
        """Returns all batches for a chunk of randomized indices for all sources."""
        counts, generator = self._source_counts(chunk)
        starts = self._start_positions(chunk)

        pool = torch.cat([
            self._take(s, int(starts[s]), int(counts[s]))
            for s in range(len(self.rows_by_source))
        ])

        pool = self.sort_pool(pool)
        batches = self.make_batches(pool)

        order = torch.randperm(len(batches), generator=generator).tolist()
        return [
            batches[i]
            for i in order
        ]

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
        """Returns random indices from a given source, seeded by the provided cycle."""
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

    def __iter__(self) -> Iterator[list[int]]:
        """Yields batches from chunks."""
        for chunk in itertools.count(self.start_chunk):
            batches = self._batches(chunk)
            yield from batches[self.start_offset if chunk == self.start_chunk else 0:]

    @abstractmethod
    def recalculate_source_probs(self) -> torch.Tensor:
        ...

    @abstractmethod
    def sort_pool(self, pool: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def make_batches(self, pool: torch.Tensor) -> list[list[int]]:
        ...
