from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from larry.voice.config.criterion.voice_criterion_configs import CTCCriterionConfig
from larry.voice.criteria.ctc import CTCCriterion
from larry.voice.model.voice_models import SpeechEncoderModelOutput
from tests.helpers.voice import ctc_output


def _output(frame_lengths: list[int]) -> SpeechEncoderModelOutput:
    generator = torch.Generator().manual_seed(0)
    log_probs = F.log_softmax(torch.randn(len(frame_lengths), max(frame_lengths), 6, generator=generator), dim=-1)
    return ctc_output(log_probs, torch.tensor(frame_lengths))


@pytest.fixture
def criterion() -> CTCCriterion:
    return CTCCriterion(CTCCriterionConfig(blank=5))


def test_sample_losses_reproduce_the_mean_loss(criterion: CTCCriterion) -> None:
    output = _output([12, 9, 7])
    targets = torch.tensor([[1, 2, 3, 4], [2, 2, 0, 0], [4, 0, 0, 0]])
    target_lengths = torch.tensor([4, 2, 1])

    sample_losses = criterion.sample_losses(output, targets, target_lengths)

    assert sample_losses.shape == (3,)
    mean_from_samples = (sample_losses / target_lengths).mean()
    assert torch.allclose(mean_from_samples, criterion(output, targets=targets, target_lengths=target_lengths))


def test_infeasible_sample_is_zeroed_not_infinite(criterion: CTCCriterion) -> None:
    output = _output([12, 2])
    targets = torch.tensor([[1, 2, 3], [1, 1, 1]])
    target_lengths = torch.tensor([3, 3])

    sample_losses = criterion.sample_losses(output, targets, target_lengths)

    assert torch.isfinite(sample_losses).all()
    assert sample_losses[1] == 0


def test_each_ctc_head_is_its_own_weighted_term_per_row() -> None:
    torch.manual_seed(0)
    log_probs = torch.randn(2, 6, 5).log_softmax(-1)
    output = SimpleNamespace(
        ctc_log_probs={"encoder2_norm": log_probs, "fsq": log_probs.roll(1, dims=-1)},
        ctc_lengths=torch.tensor([6, 5]),
        logits=log_probs,
    )
    targets, target_lengths = torch.tensor([[1, 2, 3], [2, 2, 0]]), torch.tensor([3, 2])
    criterion = CTCCriterion(CTCCriterionConfig(blank=0, ctc_layer_loss_weights={"encoder2_norm": 0.7, "fsq": 0.3}))

    terms = criterion.loss_terms(output, targets=targets, target_lengths=target_lengths)

    assert {name: term.weight for name, term in terms.items()} == {"ctc/encoder2_norm": 0.7, "ctc/fsq": 0.3}
    assert all(term.sums.shape == (2,) and term.counts.tolist() == [1.0, 1.0] for term in terms.values())
    expected = sum(
        weight * F.ctc_loss(probs.transpose(0, 1), targets, output.ctc_lengths, target_lengths, reduction="mean")
        for probs, weight in ((log_probs, 0.7), (log_probs.roll(1, dims=-1), 0.3))
    )
    assert torch.allclose(criterion(output, targets=targets, target_lengths=target_lengths), expected)
