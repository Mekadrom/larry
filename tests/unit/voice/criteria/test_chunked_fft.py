import dataclasses

import pytest
import torch

from larry.voice.criteria.spectral import MultiResolutionSTFTLoss
from larry.voice.model.vocoder import LarryVocoder
from tests.unit.voice.model.test_vocoder import TINY


@pytest.mark.parametrize("chunk_rows", [1, 3, 64])
def test_chunked_spectral_loss_matches_the_whole_batch(chunk_rows: int) -> None:
    torch.manual_seed(0)
    waveforms, estimates = torch.randn(7, 4000), torch.randn(7, 4000)
    lengths = torch.tensor([4000, 3000, 4000, 2500, 4000, 4000, 1000])

    whole = MultiResolutionSTFTLoss([256, 512])(estimates, waveforms, lengths)
    chunked = MultiResolutionSTFTLoss([256, 512], chunk_rows=chunk_rows)(estimates, waveforms, lengths)

    assert torch.allclose(whole, chunked, atol=1e-6)


@pytest.mark.parametrize("chunk_rows", [1, 2, 128])
def test_chunked_one_step_waveforms_match_the_whole_batch(chunk_rows: int) -> None:
    torch.manual_seed(0)
    whole = LarryVocoder(dataclasses.replace(TINY, istft_chunk_rows=10_000)).eval()
    chunked = LarryVocoder(dataclasses.replace(TINY, istft_chunk_rows=chunk_rows)).eval()
    chunked.load_state_dict(whole.state_dict())
    xt, velocity = torch.randn(5, 2 * whole.n_bins, 40), torch.randn(5, 2 * whole.n_bins, 40)
    t = torch.rand(5)

    with torch.no_grad():
        expected = whole.one_step_waveforms(xt, t, velocity)
        actual = chunked.one_step_waveforms(xt, t, velocity)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6)

