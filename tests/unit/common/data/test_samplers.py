import itertools

import pytest
import torch
from accelerate import Accelerator, debug_launcher
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader

from larry.common.data.samplers import ShuffledBatchSampler


def _first(sampler: ShuffledBatchSampler, n: int) -> list[list[int]]:
    return list(itertools.islice(sampler, n))


def test_each_pass_is_a_shuffle_of_every_full_batch() -> None:
    sampler = ShuffledBatchSampler(n_rows=10, batch_size=3, seed=1)

    batches = _first(sampler, 9)

    assert all(len(b) == 3 for b in batches)
    for p in range(3):
        rows = [r for b in batches[p * 3:(p + 1) * 3] for r in b]
        assert len(set(rows)) == 9
        assert set(rows) <= set(range(10))
    assert batches[:3] != batches[3:6]


@pytest.mark.parametrize("consumed", [0, 2, 3, 7, 20])
def test_start_at_resumes_exactly_where_consumption_stopped(consumed: int) -> None:
    uninterrupted = _first(ShuffledBatchSampler(n_rows=10, batch_size=3, seed=1), consumed + 10)

    resumed = ShuffledBatchSampler(n_rows=10, batch_size=3, seed=1)
    resumed.start_at(consumed)

    assert _first(resumed, 10) == uninterrupted[consumed:]


def test_dataset_smaller_than_a_batch_is_rejected() -> None:
    with pytest.raises(ValueError):
        ShuffledBatchSampler(n_rows=2, batch_size=3, seed=1)


def _shard_an_endless_sampler_across_processes() -> None:
    accelerator = Accelerator(cpu=True)
    sampler = ShuffledBatchSampler(n_rows=40, batch_size=2, seed=3)
    sampler.start_at(4)
    loader = accelerator.prepare(DataLoader(list(range(40)), batch_sampler=sampler))

    mine = [batch.tolist() for batch in itertools.islice(loader, 6)]
    everyone = accelerator.gather_for_metrics(torch.tensor(mine))

    expected = _first(ShuffledBatchSampler(n_rows=40, batch_size=2, seed=3), 16)[4:16]
    assert sorted(map(tuple, everyone.reshape(-1, 2).tolist())) == sorted(map(tuple, expected))


def test_accelerate_splits_an_endless_resumed_sampler_between_processes() -> None:
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        debug_launcher(_shard_an_endless_sampler_across_processes, num_processes=2)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
