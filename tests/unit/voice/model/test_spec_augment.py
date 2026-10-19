import torch

from larry.voice.model.speech_encoder import SpeechEncoderModel
from tests.helpers.voice import tiny_speech_encoder

spec_augment = SpeechEncoderModel._spec_augment


def _mels() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    mels = torch.rand(2, 20, 50)
    lengths = torch.tensor([50, 30])
    mels[1, :, 30:] = -1.5
    return mels, lengths


def test_no_masks_leave_the_mels_unchanged() -> None:
    mels, lengths = _mels()

    assert torch.equal(spec_augment(mels, lengths, freq_masks=0, freq_width=5, time_masks=0, time_fraction=0.1), mels)


def test_masked_cells_take_the_mean_of_the_clips_real_frames() -> None:
    mels, lengths = _mels()
    torch.manual_seed(1)

    out = spec_augment(mels, lengths, freq_masks=2, freq_width=5, time_masks=4, time_fraction=0.2)

    for b, n in enumerate(lengths.tolist()):
        changed = out[b] != mels[b]
        assert changed.any()
        assert torch.allclose(out[b][changed], mels[b, :, :n].mean().expand(int(changed.sum())))


def test_time_masks_stay_inside_each_clip() -> None:
    mels, lengths = _mels()
    torch.manual_seed(2)

    out = spec_augment(mels, lengths, freq_masks=0, freq_width=0, time_masks=10, time_fraction=0.2)

    assert torch.equal(out[1, :, 30:], mels[1, :, 30:])
    masked_frames = (out != mels).all(dim=1)
    assert int(masked_frames[0].sum()) > 0 and int(masked_frames[1, :30].sum()) > 0


def test_frequency_masks_cover_whole_bins_within_the_width() -> None:
    mels, lengths = _mels()
    torch.manual_seed(3)

    out = spec_augment(mels, lengths, freq_masks=1, freq_width=5, time_masks=0, time_fraction=0.0)

    masked_bins = (out[0] != mels[0]).all(dim=1)
    assert (out[0] != mels[0]).any(dim=1).equal(masked_bins)
    assert int(masked_bins.sum()) <= 5
    indices = masked_bins.nonzero().flatten()
    if len(indices) > 1:
        assert int(indices[-1] - indices[0]) == len(indices) - 1


def _model(n_spec_freq_masks: int = 2, n_spec_time_masks: int = 10) -> SpeechEncoderModel:
    return tiny_speech_encoder(10, n_spec_freq_masks=n_spec_freq_masks, n_spec_time_masks=n_spec_time_masks)


def _forward(model: SpeechEncoderModel) -> torch.Tensor:
    torch.manual_seed(4)
    waveforms = torch.randn(2, 16000)
    return model(waveforms, torch.tensor([16000, 12000])).logits


def test_augmentation_runs_in_training_only() -> None:
    model = _model()

    trained = _forward(model.train())
    evaluated = _forward(model.eval())

    assert not torch.allclose(trained, evaluated)
    assert torch.equal(evaluated, _forward(model.eval()))


def test_zero_masks_disable_augmentation_in_training() -> None:
    model = _model(n_spec_freq_masks=0, n_spec_time_masks=0)

    assert torch.allclose(_forward(model.train()), _forward(model.eval()), atol=1e-6)
