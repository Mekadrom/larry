import itertools

import pytest
import torch

from larry.voice.data.dataloading.voice_dataloading import AudioDurationBucketingWeightedBatchSampler

SOURCE_SIZES = [37, 11, 5]
SOURCE_SECONDS = [(2.0, 6.0), (8.0, 20.0), (1.0, 3.0)]
WEIGHTS = [0.5, 0.3, 0.2]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    source_ids, durations = [], []
    for s, (n, (low, high)) in enumerate(zip(SOURCE_SIZES, SOURCE_SECONDS)):
        source_ids.append(torch.full((n,), s, dtype=torch.long))
        durations.append(low + (high - low) * torch.rand(n, generator=generator, dtype=torch.float64))
    return torch.cat(source_ids), torch.cat(durations)


def _sampler(pool_size: int = 16, max_batch_seconds: float = 40.0) -> AudioDurationBucketingWeightedBatchSampler:
    source_ids, durations = _inputs()
    return AudioDurationBucketingWeightedBatchSampler(
        source_ids,
        durations,
        WEIGHTS,
        max_batch_seconds=max_batch_seconds,
        pool_size=pool_size,
        seed=7,
    )


def _first(sampler: AudioDurationBucketingWeightedBatchSampler, n: int) -> list[list[int]]:
    return list(itertools.islice(sampler, n))


def test_iteration_is_endless_and_repeatable() -> None:
    sampler = _sampler()

    first = _first(sampler, 200)

    assert len(first) == 200
    assert _first(sampler, 200) == first


def test_chunks_follow_each_other_without_gaps() -> None:
    sampler = _sampler()

    expected = [batch for chunk in range(6) for batch in sampler._batches(chunk)]

    assert _first(sampler, len(expected)) == expected


@pytest.mark.parametrize("consumed", [0, 1, 7, 13, 50, 123])
def test_start_at_resumes_exactly_where_consumption_stopped(consumed: int) -> None:
    uninterrupted = _first(_sampler(), consumed + 30)

    resumed = _sampler()
    resumed.start_at(consumed)

    assert _first(resumed, 30) == uninterrupted[consumed:]


def test_each_source_yields_every_row_once_before_any_repeat() -> None:
    sampler = _sampler()
    source_ids, _ = _inputs()
    seen: list[list[int]] = [[] for _ in SOURCE_SIZES]
    for chunk in range(30):
        counts, _ = sampler._source_counts(chunk)
        starts = sampler._start_positions(chunk)
        for s in range(len(SOURCE_SIZES)):
            seen[s].extend(sampler._take(s, int(starts[s]), int(counts[s])).tolist())

    for s, n in enumerate(SOURCE_SIZES):
        assert len(seen[s]) >= 2 * n
        for cycle in range(len(seen[s]) // n):
            chunk = seen[s][cycle * n:(cycle + 1) * n]
            assert sorted(chunk) == sorted(torch.nonzero(source_ids == s).flatten().tolist())


def test_time_shares_approach_weights() -> None:
    sampler = _sampler(pool_size=400, max_batch_seconds=200.0)
    source_ids, durations = _inputs()
    seconds = torch.zeros(len(SOURCE_SIZES), dtype=torch.float64)
    for batch in _first(sampler, sum(sampler._batch_count(c) for c in range(25))):
        for row in batch:
            seconds[source_ids[row]] += durations[row]

    shares = seconds / seconds.sum()
    expected = torch.tensor(WEIGHTS, dtype=torch.float64) / sum(WEIGHTS)
    assert torch.allclose(shares, expected, atol=0.03), shares


def test_batches_respect_padded_seconds() -> None:
    sampler = _sampler(pool_size=64)
    _, durations = _inputs()
    for batch in _first(sampler, 100):
        if len(batch) > 1:
            assert len(batch) * float(durations[batch].max()) <= sampler.max_batch_seconds


def test_source_without_rows_is_rejected() -> None:
    source_ids, durations = _inputs()
    with pytest.raises(ValueError):
        AudioDurationBucketingWeightedBatchSampler(source_ids, durations, WEIGHTS + [0.1], 40.0, 16, 7)
