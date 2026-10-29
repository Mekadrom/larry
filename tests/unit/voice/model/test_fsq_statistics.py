import math

import pytest
import torch

from larry.voice.model.fsq import FSQ, code_counts, codebook_statistics, fsq_codes


def test_padded_frames_are_not_counted_and_rows_go_to_their_group() -> None:
    unit_ids = torch.tensor([[0, 1, 1, 8], [2, 2, 5, 5]])
    frame_lengths = torch.tensor([3, 2])

    counts = code_counts(unit_ids, frame_lengths, codebook_size=9, group_ids=torch.tensor([1, 0]), n_groups=2)

    assert counts.tolist() == [[0, 0, 2, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0]]


def test_uniform_codes_use_the_whole_codebook() -> None:
    statistics = codebook_statistics(torch.ones(27, dtype=torch.long), levels=[3, 3, 3])

    assert statistics["usage"] == 1.0
    assert statistics["perplexity"] == pytest.approx(27)
    for d in range(3):
        assert [statistics[f"dim{d}/level{v}"] for v in (-1, 0, 1)] == pytest.approx([1 / 3] * 3)


def test_a_single_code_has_perplexity_one_and_shows_each_dimension_stuck() -> None:
    counts = torch.zeros(9, dtype=torch.long)
    counts[5] = 10

    statistics = codebook_statistics(counts, levels=[3, 3])

    assert statistics["usage"] == pytest.approx(1 / 9)
    assert statistics["perplexity"] == pytest.approx(1.0)
    assert statistics["dim0/level1"] == 1.0
    assert statistics["dim1/level0"] == 1.0


def test_level_shares_match_the_quantized_values_fsq_produced() -> None:
    torch.manual_seed(0)
    fsq = FSQ(16, [3, 3, 3, 3])
    x = torch.randn(1, 200, 16) * 3

    quantized = fsq.quantize(x)
    _, unit_ids = fsq(x)
    statistics = codebook_statistics(code_counts(unit_ids, torch.tensor([200]), 81)[0], levels=[3, 3, 3, 3])

    for d in range(4):
        for v in (-1, 0, 1):
            expected = float((quantized[0, :, d].round() == v).double().mean())
            assert math.isclose(statistics[f"dim{d}/level{v}"], expected, abs_tol=1e-9)


def test_no_frames_gives_no_statistics() -> None:
    assert codebook_statistics(torch.zeros(9, dtype=torch.long), levels=[3, 3]) == {}


@pytest.mark.parametrize("levels", [[8, 8, 8, 6, 5], [8, 5, 5, 5], [3, 3]])
def test_mixed_even_and_odd_levels_use_every_code_once(levels: list[int]) -> None:
    fsq = FSQ(8, levels)
    n_codes = math.prod(levels)
    digits = torch.stack(torch.meshgrid(*[torch.arange(level) for level in levels], indexing="ij"), dim=-1)
    q = digits.reshape(-1, len(levels)).float() - fsq.half_width

    assert sorted(fsq.indices(q).tolist()) == list(range(n_codes))
    assert torch.equal(fsq_codes(fsq.indices(q), levels), q / fsq.half_width)


@pytest.mark.parametrize("levels", [[8, 8, 8, 6, 5], [4, 4]])
def test_quantized_values_stay_within_each_dimensions_levels(levels: list[int]) -> None:
    torch.manual_seed(0)
    fsq = FSQ(16, levels)

    q = fsq.quantize(torch.randn(4096, 16) * 50)

    for d, level in enumerate(levels):
        values = set(q[:, d].round().tolist())
        assert values <= set(range(-(level // 2), level - level // 2))
        assert len(values) == level


def test_even_level_statistics_are_labelled_from_their_lowest_value() -> None:
    statistics = codebook_statistics(torch.ones(4, dtype=torch.long), levels=[4])

    assert [statistics[f"dim0/level{v}"] for v in (-2, -1, 0, 1)] == pytest.approx([0.25] * 4)
