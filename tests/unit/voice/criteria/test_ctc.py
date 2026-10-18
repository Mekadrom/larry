
import torch
import torch.nn.functional as F

from larry.common.config.criteria.criterion_configs import CTCCriterionConfig
from larry.common.model.model import LarrySpeechEncoderModelOutput
from larry.voice.criteria.ctc import CTCCriterion


def _output(frame_lengths: list[int]) -> LarrySpeechEncoderModelOutput:
    generator = torch.Generator().manual_seed(0)
    log_probs = F.log_softmax(torch.randn(len(frame_lengths), max(frame_lengths), 6, generator=generator), dim=-1)
    return LarrySpeechEncoderModelOutput(
        logits=log_probs,
        ctc_log_probs={"encoder2_norm": log_probs},
        frame_lengths=torch.tensor(frame_lengths),
        unit_ids=torch.zeros(len(frame_lengths), max(frame_lengths), dtype=torch.long),
    )


def test_sample_losses_reproduce_the_mean_loss() -> None:
    criterion = CTCCriterion(CTCCriterionConfig(blank=5))
    output = _output([12, 9, 7])
    targets = torch.tensor([[1, 2, 3, 4], [2, 2, 0, 0], [4, 0, 0, 0]])
    target_lengths = torch.tensor([4, 2, 1])

    sample_losses = criterion.sample_losses(output, targets, target_lengths)

    assert sample_losses.shape == (3,)
    mean_from_samples = (sample_losses / target_lengths).mean()
    assert torch.allclose(mean_from_samples, criterion(output, targets=targets, target_lengths=target_lengths))


def test_infeasible_sample_is_zeroed_not_infinite() -> None:
    criterion = CTCCriterion(CTCCriterionConfig(blank=5))
    output = _output([12, 2])
    targets = torch.tensor([[1, 2, 3], [1, 1, 1]])
    target_lengths = torch.tensor([3, 3])

    sample_losses = criterion.sample_losses(output, targets, target_lengths)

    assert torch.isfinite(sample_losses).all()
    assert sample_losses[1] == 0
