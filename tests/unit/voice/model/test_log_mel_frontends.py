import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from larry.voice.model.voice_modules import ASRLogMelFrontend, TTSLogMelFrontend
from larry.voice.utils import bytes_to_waveforms
from tests.helpers.audio import wav

FRONTENDS = {
    "asr": lambda: ASRLogMelFrontend(16000, 100, 400, 160),
    "tts": lambda: TTSLogMelFrontend(16000, 80, 640, 160),
}


def _clips(seconds: list[float]) -> list[torch.Tensor]:
    return [
        bytes_to_waveforms(wav(s, 16000, frequency=150 + 70 * i, gate=700 + 31 * i), 16000)[0]
        for i, s in enumerate(seconds)
    ]


@pytest.mark.parametrize("name", FRONTENDS)
def test_valid_frames_do_not_depend_on_the_rest_of_the_batch(name: str) -> None:
    frontend = FRONTENDS[name]()
    clips = _clips([2.3, 0.41, 1.07, 0.3])
    lengths = torch.tensor([len(c) for c in clips])

    batched, batched_lengths = frontend(pad_sequence(clips, batch_first=True), lengths)

    for i, clip in enumerate(clips):
        alone, alone_lengths = frontend(clip[None], lengths[i:i + 1])
        n = int(alone_lengths[0])
        assert int(batched_lengths[i]) == n
        torch.testing.assert_close(batched[i, :, :n], alone[0, :, :n], rtol=0, atol=0)


@pytest.mark.parametrize("name", FRONTENDS)
def test_counted_frames_have_windows_inside_the_clip(name: str) -> None:
    frontend = FRONTENDS[name]()
    half_window, hop = frontend.mel.n_fft // 2, frontend.hop_length
    samples = torch.tensor([0, half_window - 1, half_window, half_window + hop, 8000])

    _, lengths = frontend(torch.zeros(len(samples), 8000), samples)

    assert lengths.tolist() == [0, 0, 1, 2, (8000 - half_window) // hop + 1]
    assert ((lengths - 1) * hop + half_window <= samples.clamp(min=half_window)).all()
