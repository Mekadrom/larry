import dataclasses
import math
from typing import Literal

import torch
from torch import nn


@dataclasses.dataclass(kw_only=True)
class F0Statistics:
    log_f0: torch.Tensor
    voiced: torch.Tensor
    contour: torch.Tensor
    log_f0_mean: torch.Tensor
    log_f0_spread: torch.Tensor
    has_range: torch.Tensor

    @classmethod
    def from_f0(cls, f0: torch.Tensor, frame_mask: torch.Tensor, min_spread: float = 1e-3) -> F0Statistics:
        voiced = (f0 > 0) & frame_mask.bool()
        log_f0 = torch.where(voiced, f0.clamp(min=1.0).log(), torch.zeros_like(f0))
        counts = voiced.sum(dim=1)
        has_range = counts >= 2
        mean = log_f0.sum(dim=1) / counts.clamp(min=1)
        centered = torch.where(voiced, log_f0 - mean[:, None], torch.zeros_like(log_f0))
        raw_spread = (centered.pow(2).sum(dim=1) / counts.clamp(min=1)).sqrt()
        spread = raw_spread.clamp(min=min_spread)
        varies = has_range & (raw_spread > min_spread)
        contour = torch.where(voiced & varies[:, None], centered / spread[:, None], torch.zeros_like(log_f0))
        return cls(
            log_f0=log_f0,
            voiced=voiced.to(f0.dtype),
            contour=contour,
            log_f0_mean=mean,
            log_f0_spread=spread,
            has_range=has_range,
        )


class F0RangePredictor(nn.Module):
    def __init__(self, speaker_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(speaker_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        out = self.net[-1]
        assert isinstance(out, nn.Linear)
        nn.init.zeros_(out.weight)
        out.bias.copy_(torch.tensor([math.log(150.0), math.log(0.2)]))

    def forward(self, speaker_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_spread = self.net(speaker_embeddings).float().unbind(-1)
        return mean, log_spread


class HarmonicConditioning(nn.Module):
    def __init__(
            self,
            mode: Literal["comb", "sinusoid"],
            mel_filterbank: torch.Tensor,
            sample_rate: int,
            hop_length: int,
            n_harmonics: int,
            comb_sharpness: float,
            n_aperiodicity_bands: int | None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_harmonics = n_harmonics
        self.comb_sharpness = comb_sharpness
        self.frame_seconds = hop_length / sample_rate
        self.n_aperiodicity_bands = n_aperiodicity_bands
        n_freqs, n_mels = mel_filterbank.shape
        weights = mel_filterbank / mel_filterbank.sum(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer("mel_weights", weights.float(), persistent=False)
        self.register_buffer("frequencies", torch.linspace(0, sample_rate / 2, n_freqs), persistent=False)
        self.harmonic_channels = n_mels if mode == "comb" else 2 * n_harmonics

    mel_weights: torch.Tensor
    frequencies: torch.Tensor

    @property
    def channels(self) -> int:
        return 3 + self.harmonic_channels + (self.n_aperiodicity_bands or 0)

    def comb(self, log_f0: torch.Tensor) -> torch.Tensor:
        f0 = log_f0.float().exp()
        ratio = self.frequencies / f0.unsqueeze(-1)
        peaks = ((1 + torch.cos(2 * math.pi * ratio)) / 2).pow(self.comb_sharpness) * (ratio >= 0.5)
        return peaks @ self.mel_weights

    def sinusoids(self, log_f0: torch.Tensor) -> torch.Tensor:
        cumulative_phase = (2 * math.pi * self.frame_seconds) * torch.cumsum(log_f0.float().exp(), dim=1)
        features = []
        for h in range(1, self.n_harmonics + 1):
            phase = torch.remainder(h * cumulative_phase, 2 * math.pi)
            features += [torch.sin(phase), torch.cos(phase)]

        return torch.stack(features, dim=-1)

    def forward(
            self,
            contour: torch.Tensor,
            voiced: torch.Tensor,
            log_f0: torch.Tensor,
            aperiodicity: torch.Tensor | None,
    ) -> torch.Tensor:
        harmonics = self.comb(log_f0) if self.mode == "comb" else self.sinusoids(log_f0)
        harmonics = harmonics * voiced.unsqueeze(-1)
        scalars = torch.stack([contour, voiced, voiced * (log_f0 - math.log(150.0))], dim=-1)
        parts = [scalars.float(), harmonics]
        if self.n_aperiodicity_bands is not None:
            if aperiodicity is None:
                aperiodicity = torch.zeros(*contour.shape, self.n_aperiodicity_bands, device=contour.device)

            parts.append(aperiodicity.float())

        return torch.cat(parts, dim=-1).transpose(1, 2)
