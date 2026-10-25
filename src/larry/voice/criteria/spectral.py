import torch
from torch import nn

from larry.common.model.masks import make_non_pad_mask


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: list[int], eps: float = 1e-5) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps
        for n_fft in fft_sizes:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))

    def magnitudes(self, waveforms: torch.Tensor, n_fft: int) -> torch.Tensor:
        window = self.get_buffer(f"window_{n_fft}")
        return torch.stft(
            waveforms.float(),
            n_fft,
            n_fft // 4,
            window=window,
            center=True,
            return_complex=True,
        ).abs()

    def resolution_loss(
            self,
            estimates: torch.Tensor,
            waveforms: torch.Tensor,
            waveform_lengths: torch.Tensor,
            n_fft: int,
    ) -> torch.Tensor:
        estimate, real = self.magnitudes(estimates, n_fft), self.magnitudes(waveforms, n_fft)
        mask = make_non_pad_mask(waveform_lengths // (n_fft // 4) + 1, real.shape[-1]).unsqueeze(1).to(real.dtype)
        convergence = ((real - estimate) * mask).square().sum(dim=(1, 2)).sqrt() / (
            (real * mask).square().sum(dim=(1, 2)).sqrt().clamp(min=self.eps)
        )
        log_distance = ((real + self.eps).log() - (estimate + self.eps).log()).abs() * mask
        log_magnitude = log_distance.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * real.shape[1]).clamp(min=1)
        return convergence + log_magnitude

    def forward(self, estimates: torch.Tensor, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            losses = [
                self.resolution_loss(estimates, waveforms, waveform_lengths, n_fft) for n_fft in self.fft_sizes
            ]
            
        return torch.stack(losses).mean(dim=0)
