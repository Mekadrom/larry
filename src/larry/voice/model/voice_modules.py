import math
from typing import cast

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from larry.common.model.masks import make_non_pad_mask
from larry.voice.config.training.voice_trainer_configs import MelAugmentConfig


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


class MelAugmenter:
    def __init__(self, config: MelAugmentConfig) -> None:
        self.config = config

    @staticmethod
    def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
        radius = max(1, math.ceil(3 * sigma))
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        return kernel / kernel.sum()

    def blur_last(self, mels: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel = self.gaussian_kernel(sigma, mels.device)
        radius = kernel.shape[0] // 2
        flat = F.pad(mels.reshape(-1, 1, mels.shape[-1]), (radius, radius), mode="replicate")
        return F.conv1d(flat, kernel.view(1, 1, -1)).view(mels.shape)

    @staticmethod
    def chosen(p: float, mels: torch.Tensor) -> torch.Tensor:
        return torch.rand(mels.shape[0], 1, 1, device=mels.device) < p

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        config = self.config
        if config.time_blur_p > 0:
            blurred = self.blur_last(mels, config.time_blur_sigma)
            mels = torch.where(self.chosen(config.time_blur_p, mels), blurred, mels)

        if config.freq_blur_p > 0:
            blurred = self.blur_last(mels.transpose(1, 2), config.freq_blur_sigma).transpose(1, 2)
            mels = torch.where(self.chosen(config.freq_blur_p, mels), blurred, mels)

        if config.noise_p > 0:
            std = torch.rand(mels.shape[0], 1, 1, device=mels.device) * config.noise_std
            mels = mels + torch.randn_like(mels) * std * self.chosen(config.noise_p, mels)

        return mels
