import dataclasses

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.model.flow import ConditionalFlowMatching, LarryFlowMatchingOutput, TimestepEmbedding
from larry.common.model.norms import AdaLayerNorm, AdaLayerNormZero
from larry.common.training.training_model import GenerationOutput
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.model.voice_models import VoiceModel
from larry.voice.model.voice_modules import TTSLogMelFrontend
from larry.voice.utils import make_non_pad_mask


@dataclasses.dataclass(kw_only=True)
class LarryVocoderGeneration(GenerationOutput):
    waveforms: torch.Tensor
    waveform_lengths: torch.Tensor


class CompressedSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, alpha: float, scale: float) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.scale = scale
        self.register_buffer("window", torch.hann_window(n_fft))

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(waveforms, self.n_fft, self.hop_length, window=self.window, center=True, return_complex=True)
        spec = spec.abs().clamp(min=1e-8).pow(self.alpha) * torch.exp(1j * spec.angle())
        return torch.cat([spec.real, spec.imag], dim=1) * self.scale

    def decode(self, x: torch.Tensor, length: int) -> torch.Tensor:
        real, imag = (x / self.scale).chunk(2, dim=1)
        spec = torch.complex(real, imag)
        spec = spec.abs().clamp(min=1e-8).pow(1 / self.alpha) * torch.exp(1j * spec.angle())
        return torch.istft(spec, self.n_fft, self.hop_length, window=self.window, center=True, length=length)


class ConditionalConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = AdaLayerNormZero(dim, cond_dim)
        self.pointwise1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pointwise2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(x * mask.unsqueeze(1)).transpose(1, 2)
        h, gate = self.norm(h, cond)
        h = self.pointwise2(self.act(self.pointwise1(h)))
        return x + (gate * h).transpose(1, 2)


class VocoderVelocityNetwork(nn.Module):
    def __init__(self, n_mels: int, n_bins: int, d_model: int, d_inner: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Conv1d(n_mels + 2 * n_bins, d_model, kernel_size=7, padding=3)
        self.time = TimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([
            ConditionalConvNeXtBlock(d_model, d_inner, d_model) for _ in range(n_layers)
        ])
        self.final_norm = AdaLayerNorm(d_model, d_model)
        self.out = nn.Linear(d_model, 2 * n_bins)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, mels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cond = self.time(t)
        x = self.embed(torch.cat([xt, mels], dim=1) * mask.unsqueeze(1))
        for block in self.blocks:
            x = block(x, cond, mask)

        return self.out(self.final_norm(x.transpose(-2, -1), cond)).transpose(-2, -1) * mask.unsqueeze(1)


class LarryVocoder(VoiceModel[LarryVocoderConfig], PyTorchModelHubMixin, ConditionalFlowMatching):
    def __init__(self, config: LarryVocoderConfig) -> None:
        VoiceModel.__init__(self, config)
        ConditionalFlowMatching.__init__(self, config.sigma_min)

        self.n_bins = self.config.n_fft // 2 + 1
        self.frontend = TTSLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
        self.codec = CompressedSTFT(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            alpha=config.stft_alpha,
            scale=config.stft_scale
        )
        self.backbone = VocoderVelocityNetwork(
            n_mels=config.n_mels,
            n_bins=self.n_bins,
            d_model=config.d_model,
            d_inner=config.d_inner,
            n_layers=config.n_layers
        )

    def forward(self, spec: torch.Tensor, mels: torch.Tensor, mel_lengths: torch.Tensor) -> LarryFlowMatchingOutput:
        mask = make_non_pad_mask(mel_lengths, spec.shape[-1]).to(spec.dtype)
        xt, t, target = self.path(spec)
        pred = self.backbone(xt, t, mels, mask)
        return LarryFlowMatchingOutput(logits=pred, targets=target, mask=mask)

    def generate(
            self,
            mels: torch.Tensor,
            mel_lengths: torch.Tensor,
            steps: int,
            generator: torch.Generator | None = None,
    ) -> LarryVocoderGeneration:
        mask = make_non_pad_mask(mel_lengths, mels.shape[-1]).to(mels.dtype)
        shape = (mels.shape[0], self.config.n_fft + 2, mels.shape[-1])
        spec = self.sample(lambda x, t: self.backbone(x, t, mels, mask), shape, steps, self.device, generator)
        waveforms = self.codec.decode(spec, (mels.shape[-1] - 1) * self.config.hop_length)
        lengths = (mel_lengths * self.config.hop_length - self.config.n_fft // 2).clamp(min=0)
        return LarryVocoderGeneration(waveforms=waveforms, waveform_lengths=lengths)
