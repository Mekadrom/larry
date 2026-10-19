import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler[list[int]], ABC):
    """Super class for training-time batch samplers that supports resuming from a batch index."""

    @abstractmethod
    def start_at(self, consumed_batches: int) -> None:
        """Starts the sampler at the specified index (a count of consumed batches)."""
        ...

    @staticmethod
    def seeded_generator(*keys: int) -> torch.Generator:
        """A generator whose stream is unique to ``keys``, unlike ``seed + k`` where adjacent seeds share streams."""
        seed = int(np.random.SeedSequence(list(keys)).generate_state(1)[0])
        return torch.Generator().manual_seed(seed)


class ShuffledBatchSampler(ResumableBatchSampler):
    """An implementation of the ``ResumableBatchSampler`` that shuffles indices before returning batches."""

    def __init__(self, n_rows: int, batch_size: int, seed: int) -> None:
        """Sets up this instance of ``ShuffledBatchSampler`` with initial common sense validation."""
        if n_rows < batch_size:
            raise ValueError(f"n_rows={n_rows} is smaller than batch_size={batch_size}")

        self.n_rows = n_rows
        self.batch_size = batch_size
        self.seed = seed
        self.batches_per_pass = n_rows // batch_size
        self.consumed_batches = 0

    def start_at(self, consumed_batches: int) -> None:
        """Simple resume override."""
        self.consumed_batches = consumed_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yields shuffled batches."""
        first_pass, offset = divmod(self.consumed_batches, self.batches_per_pass)
        for p in itertools.count(first_pass):
            generator = self.seeded_generator(self.seed, p)
            order = torch.randperm(self.n_rows, generator=generator)
            for b in range(offset if p == first_pass else 0, self.batches_per_pass):
                yield order[b * self.batch_size:(b + 1) * self.batch_size].tolist()
