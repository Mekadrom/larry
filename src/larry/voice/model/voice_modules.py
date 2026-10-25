from typing import cast

import torch
import torchaudio
from torch import nn

from larry.common.model.masks import make_non_pad_mask


class ASRLogMelFrontend(nn.Module):
    """Just a module wrapping the ``MelSpectrogram`` transform with the correct configuration."""

    def __init__(self, sample_rate: int, n_mels: int, n_fft: int, hop_length: int) -> None:
        """Initializes the delegate transform and window cache."""
        super().__init__()
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            mel_scale="slaney",
        )
        # the periodic window is a slice of a longer one; safetensors refuses to save a buffer that shares storage
        self.mel.spectrogram.window = cast(torch.Tensor, self.mel.spectrogram.window).clone()

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the mels without casting to lower precision."""
        mel_lengths = ((waveform_lengths - self.mel.n_fft // 2) // self.hop_length + 1).clamp(min=0)
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            mels = self.mel(waveforms.float()).clamp(min=1e-10).log10()  # [B, n_mels, T]
            # whisper-style norm
            valid = make_non_pad_mask(mel_lengths, mels.shape[-1]).unsqueeze(1)
            peak = mels.masked_fill(~valid, float("-inf")).amax(dim=(1, 2), keepdim=True)
            mels = torch.maximum(mels, peak - 8.0)
            mels = (mels + 4.0) / 4.0

        return mels, mel_lengths


class TTSLogMelFrontend(nn.Module):
    """Just a module wrapping the ``MelSpectrogram`` transform with the correct configuration."""

    def __init__(self, sample_rate: int, n_mels: int, n_fft: int, hop_length: int) -> None:
        """Initializes the delegate transform and window cache."""
        super().__init__()
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            mel_scale="slaney",
        )
        # the periodic window is a slice of a longer one; safetensors refuses to save a buffer that shares storage
        self.mel.spectrogram.window = cast(torch.Tensor, self.mel.spectrogram.window).clone()

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the mels without casting to lower precision."""
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            mels = self.mel(waveforms.float()).clamp(min=1e-5).log10()  # [B, n_mels, T]

        mel_lengths = ((waveform_lengths - self.mel.n_fft // 2) // self.hop_length + 1).clamp(min=0)
        return mels, mel_lengths
