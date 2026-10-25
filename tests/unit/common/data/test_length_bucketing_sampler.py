import pytest
import torch

from larry.common.data.dataloading.weighted_batch_sampler import (
    FixedSizeWeightedBatchSampler,
    LengthBucketingWeightedBatchSampler,
    length_batches,
)
from tests.helpers.common import first


def test_length_batches_keep_the_padded_length_within_budget() -> None:
    lengths = torch.tensor([3, 5, 5, 9, 20, 40])

    batches = length_batches(list(range(6)), lengths, 20)

    assert batches == [[0, 1, 2], [3], [4], [5]]
    assert all(len(b) * int(lengths[b].max()) <= 20 or len(b) == 1 for b in batches)


def test_weights_are_shares_of_length_not_rows() -> None:
    source_ids = torch.tensor([0] * 100 + [1] * 100)
    lengths = torch.cat([torch.full((100,), 10), torch.full((100,), 40)])
    sampler = LengthBucketingWeightedBatchSampler(source_ids, lengths, [1.0, 1.0], 400, pool_size=500, seed=3)

    rows = torch.tensor([row for batch in first(sampler, 400) for row in batch])

    short, long = (rows < 100).sum(), (rows >= 100).sum()
    assert float(short * 10) / float(long * 40) == pytest.approx(1.0, rel=0.1)


def test_resuming_continues_the_same_stream() -> None:
    source_ids = torch.tensor([0] * 30 + [1] * 30)
    lengths = torch.arange(60) % 17 + 1
    batches = first(LengthBucketingWeightedBatchSampler(source_ids, lengths, [2.0, 1.0], 64, 32, 5), 60)
    resumed = LengthBucketingWeightedBatchSampler(source_ids, lengths, [2.0, 1.0], 64, 32, 5)

    resumed.start_at(25)

    assert first(resumed, 20) == batches[25:45]


def test_the_sampler_mixes_sources_by_weight_and_resumes() -> None:
    source_ids = torch.tensor([0] * 50 + [1] * 50)
    sampler = FixedSizeWeightedBatchSampler(source_ids, [3.0, 1.0], batch_size=4, pool_size=400, seed=1)

    batches = [batch for batch, _ in zip(sampler, range(100), strict=False)]
    resumed = FixedSizeWeightedBatchSampler(source_ids, [3.0, 1.0], batch_size=4, pool_size=400, seed=1)
    resumed.start_at(40)

    rows = torch.tensor([row for batch in batches for row in batch])
    assert all(len(batch) == 4 for batch in batches)
    assert float((rows < 50).float().mean()) == pytest.approx(0.75, abs=0.06)
    assert any(len({row < 50 for row in batch}) == 2 for batch in batches)
    assert [batch for batch, _ in zip(resumed, range(10), strict=False)] == batches[40:50]
