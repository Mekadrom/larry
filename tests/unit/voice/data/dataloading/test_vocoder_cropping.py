from types import SimpleNamespace

import pytest
import torch

from larry.voice.data.dataloading.voice_dataloading import (
    ClipCropWeightedBatchSampler,
    RandomCropCollator,
    WaveformCollator,
)
from larry.voice.training.voice_trainers import VocoderTrainer
from tests.helpers.common import first, uninitialized

SOURCE_SIZES = [40, 25, 12]
SOURCE_SECONDS = [(0.5, 3.0), (6.0, 20.0), (2.0, 9.0)]
WEIGHTS = [0.5, 0.3, 0.2]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    source_ids, durations = [], []
    for s, (n, (low, high)) in enumerate(zip(SOURCE_SIZES, SOURCE_SECONDS, strict=True)):
        source_ids.append(torch.full((n,), s, dtype=torch.long))
        durations.append(low + (high - low) * torch.rand(n, generator=generator, dtype=torch.float64))
    return torch.cat(source_ids), torch.cat(durations)


def _sampler(clips_per_batch: int = 8, pool_size: int = 64, seed: int = 3) -> ClipCropWeightedBatchSampler:
    source_ids, durations = _inputs()
    return ClipCropWeightedBatchSampler(
        source_ids,
        durations,
        WEIGHTS,
        clips_per_batch=clips_per_batch,
        crop_seconds=2.0,
        crops_per_clip=4,
        pool_size=pool_size,
        seed=seed,
    )


def _items(lengths: list[int]) -> list[dict[str, object]]:
    return [{"waveform": torch.arange(n, dtype=torch.float32) + 1, "source": i % 2} for i, n in enumerate(lengths)]


@pytest.mark.parametrize("n_samples", [100, 160, 161, 320, 333, 639, 640, 641, 5000])
def test_crops_are_in_bounds_and_do_not_overlap(n_samples: int) -> None:
    collator = RandomCropCollator(crop_samples=160, crops_per_clip=4, seed=0)

    starts = collator.crop_starts(n_samples)

    if n_samples <= 160:
        assert starts == [0]
        return
    assert len(starts) == min(4, n_samples // 160)
    assert starts == sorted(starts)
    assert all(s >= 0 and s + 160 <= n_samples for s in starts)
    assert all(b - a >= 160 for a, b in zip(starts, starts[1:], strict=False))


def test_collated_crops_have_fixed_width_and_real_lengths() -> None:
    collator = RandomCropCollator(crop_samples=160, crops_per_clip=3, seed=0)
    items = _items([1000, 90, 400])

    batch = collator(items)

    assert batch["waveforms"].shape == (3 + 1 + 2, 160)
    assert batch["waveform_lengths"].tolist() == [160, 160, 160, 90, 160, 160]
    assert batch["source_ids"].tolist() == [0, 0, 0, 1, 0, 0]
    assert torch.equal(batch["waveforms"][3, :90], items[1]["waveform"])
    assert not batch["waveforms"][3, 90:].any()


def test_each_crop_is_a_contiguous_slice_of_its_clip() -> None:
    collator = RandomCropCollator(crop_samples=160, crops_per_clip=4, seed=5)

    batch = collator(_items([2000]))

    for crop in batch["waveforms"]:
        assert torch.equal(crop, torch.arange(int(crop[0]), int(crop[0]) + 160, dtype=torch.float32))


def test_crops_follow_the_seed_and_the_process() -> None:
    def starts(seed: int, process_index: int) -> list[list[int]]:
        collator = RandomCropCollator(crop_samples=160, crops_per_clip=4, seed=seed, process_index=process_index)
        return [collator.crop_starts(5000) for _ in range(5)]

    assert starts(1, 0) == starts(1, 0)
    assert starts(1, 0) != starts(2, 0)
    assert starts(1, 0) != starts(1, 1)


def test_waveform_collator_pads_full_clips() -> None:
    batch = WaveformCollator()(_items([5, 3]))

    assert batch["waveforms"].shape == (2, 5)
    assert batch["waveform_lengths"].tolist() == [5, 3]
    assert batch["source_ids"].tolist() == [0, 1]


def test_sampler_batches_hold_a_fixed_number_of_clips() -> None:
    batches = first(_sampler(clips_per_batch=8, pool_size=64), 40)

    assert all(len(b) == 8 for b in batches)


def test_sampler_mixes_sources_within_batches() -> None:
    source_ids, _ = _inputs()

    batches = first(_sampler(clips_per_batch=8, pool_size=64), 40)

    assert sum(len(set(source_ids[b].tolist())) > 1 for b in batches) > 30


def test_sampler_is_repeatable_and_resumable() -> None:
    expected = first(_sampler(), 30)
    resumed = _sampler()
    resumed.start_at(11)

    assert first(_sampler(), 30) == expected
    assert first(resumed, 19) == expected[11:]


def test_cropped_time_shares_approach_weights() -> None:
    sampler = _sampler(clips_per_batch=16, pool_size=512, seed=11)
    source_ids, durations = _inputs()
    seconds = torch.zeros(len(WEIGHTS), dtype=torch.float64)

    for batch in first(sampler, 600):
        rows = torch.tensor(batch)
        seconds.index_add_(0, source_ids[rows], sampler.cropped_seconds(durations[rows]))

    shares = seconds / seconds.sum()
    assert torch.allclose(shares, torch.tensor(WEIGHTS, dtype=torch.float64), atol=0.03)


def test_cropped_seconds_match_what_the_collator_takes() -> None:
    sampler = _sampler()
    collator = RandomCropCollator(crop_samples=32_000, crops_per_clip=4, seed=0)
    durations = torch.tensor([0.5, 2.0, 3.9, 4.0, 7.5, 30.0], dtype=torch.float64)

    taken = [
        sum(min(32_000, n - s) for s in collator.crop_starts(n)) / 16_000
        for n in (durations * 16_000).long().tolist()
    ]

    assert sampler.cropped_seconds(durations).tolist() == pytest.approx(taken)


def _trainer() -> VocoderTrainer:
    return uninitialized(VocoderTrainer, _config=SimpleNamespace(
        crop_seconds=2.0,
        crops_per_clip=4,
        clips_per_batch=16,
        seed=9,
        mixture_config=SimpleNamespace(sample_rate=16_000, pool_size=128, seed=1, max_batch_seconds=30.0),
    ))


def test_trainer_crops_for_training_and_keeps_full_clips_for_eval() -> None:
    trainer = _trainer()

    train = trainer.make_collate_fn()
    evaluate = trainer.make_eval_collate_fn()

    assert isinstance(train, RandomCropCollator)
    assert train.crop_samples == 32_000
    assert train.crops_per_clip == 4
    assert isinstance(evaluate, WaveformCollator)


def test_trainer_samples_clips_by_cropped_time() -> None:
    source_ids, durations = _inputs()
    dataset = SimpleNamespace(source_ids_and_durations=lambda: (source_ids, durations))

    sampler = _trainer().make_train_sampler(dataset, WEIGHTS)

    assert isinstance(sampler, ClipCropWeightedBatchSampler)
    assert all(len(b) == 16 for b in first(sampler, 10))


def test_trainer_eval_batches_cover_every_clip_once() -> None:
    _, durations = _inputs()
    dataset = SimpleNamespace(source_ids_and_durations=lambda: (None, durations))

    batches = _trainer().make_eval_sampler(dataset)

    assert sorted(r for b in batches for r in b) == list(range(len(durations)))


def test_vocoder_eval_losses_are_grouped_by_source() -> None:
    from larry.common.config.criteria.criterion_configs import CriterionConfig
    from larry.common.criteria.flow_matching import ConditionalFlowMatchingCriterion
    from larry.common.model.flow import LarryFlowMatchingOutput
    from larry.common.training.trainers import StepOutput

    trainer = uninitialized(
        VocoderTrainer,
        _config=SimpleNamespace(
            mixture_config=SimpleNamespace(eval_sources=[SimpleNamespace(name="a"), SimpleNamespace(name="b")])
        ),
        criterion=ConditionalFlowMatchingCriterion(CriterionConfig()),
    )
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    output = LarryFlowMatchingOutput(logits=torch.ones(2, 4, 3), targets=torch.zeros(2, 4, 3), mask=mask)
    sample_losses = trainer.criterion.sample_losses(output, output.targets)

    groups = trainer.eval_group_totals(
        {"source_ids": torch.tensor([1, 0])},
        StepOutput(loss=torch.tensor(1.0), metrics=None, sample_losses=sample_losses, model_output=output),
    )

    assert trainer.eval_group_names() == ["a", "b"]
    assert groups[0].tolist() == [1, 0]
    assert groups[1].tolist() == [8.0, 4.0]
    assert groups[2].tolist() == [8.0, 4.0]
