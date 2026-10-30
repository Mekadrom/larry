import dataclasses
import math

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.model.flow import ConditionalFlowMatching, TimestepEmbedding
from larry.common.model.init import WeightInitializer
from larry.common.model.masks import make_non_pad_mask
from larry.common.model.norms import AdaLayerNorm, AdaLayerNormZero
from larry.common.training.training_model import GenerationOutput
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.model.voice_models import LarryVocoderOutput, VoiceModel
from larry.voice.model.voice_modules import TTSLogMelFrontend


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
        self.register_buffer("window", torch.hann_window(n_fft).clone())

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waveforms,
            self.n_fft,
            self.hop_length,
            window=self.window,  # pyright: ignore[reportArgumentType]
            center=True,
            return_complex=True
        )
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
        ConditionalFlowMatching.__init__(self, config)

        if config.hop_length % config.stft_hop_length:
            raise ValueError(
                f"hop_length={config.hop_length} must be a multiple of stft_hop_length={config.stft_hop_length}"
            )

        self.n_bins = self.config.stft_n_fft // 2 + 1
        self.mel_upsample = config.hop_length // config.stft_hop_length
        self.frontend = TTSLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
        self.codec = CompressedSTFT(
            n_fft=config.stft_n_fft,
            hop_length=config.stft_hop_length,
            alpha=config.stft_alpha,
            scale=config.stft_scale
        )
        self.mel_upsampler = None
        if self.mel_upsample > 1:
            self.mel_upsampler = nn.ConvTranspose1d(
                config.n_mels,
                config.n_mels,
                kernel_size=2 * self.mel_upsample,
                stride=self.mel_upsample,
                padding=self.mel_upsample // 2,
            )
        self.backbone = VocoderVelocityNetwork(
            n_mels=config.n_mels,
            n_bins=self.n_bins,
            d_model=config.d_model,
            d_inner=config.d_inner,
            n_layers=config.n_layers
        )
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        WeightInitializer("trunc_normal", 0.02).apply(self)
        nn.init.zeros_(self.backbone.out.weight)
        nn.init.zeros_(self.backbone.out.bias)

    def upsample_mels(self, mels: torch.Tensor, frames: int) -> torch.Tensor:
        if self.mel_upsampler is None:
            return mels[..., :frames]

        upsampled = self.mel_upsampler(mels)[..., :frames]
        return F.pad(upsampled, (0, frames - upsampled.shape[-1]))

    @property
    def receptive_field(self) -> int:
        stft_frames = 3 * (len(self.backbone.blocks) + 1) + self.config.stft_n_fft // (2 * self.config.stft_hop_length)
        return math.ceil(stft_frames / self.mel_upsample) + 1

    def stream(
            self,
            steps: int,
            lookahead: int | None = None,
            left_context: int | None = None,
            crossfade: int | None = None,
            generator: torch.Generator | None = None,
            sway_coefficient: float | None = None,
    ) -> LarryVocoderStream:
        return LarryVocoderStream(self, steps, lookahead, left_context, crossfade, generator, sway_coefficient)

    def spec_lengths(self, mel_lengths: torch.Tensor, frames: int) -> torch.Tensor:
        return (mel_lengths * self.mel_upsample).clamp(max=frames)

    def forward(self, spec: torch.Tensor, mels: torch.Tensor, mel_lengths: torch.Tensor) -> LarryVocoderOutput:
        frames = spec.shape[-1]
        mask = make_non_pad_mask(self.spec_lengths(mel_lengths, frames), frames).to(spec.dtype)
        xt, t, target = self.path(spec)
        pred = self.backbone(xt, t, self.upsample_mels(mels, frames), mask)
        return LarryVocoderOutput(
            logits=pred,
            targets=target,
            mask=mask,
            t=t,
            waveform_estimates=self.one_step_waveforms(xt, t, pred),
        )

    def one_step_waveforms(self, xt: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=xt.device.type, enabled=False):
            spec = xt.float() + (1 - t.float()).view(-1, 1, 1) * velocity.float()
            return self.codec.decode(spec, (spec.shape[-1] - 1) * self.config.stft_hop_length)

    def generate(
            self,
            mels: torch.Tensor,
            mel_lengths: torch.Tensor,
            steps: int,
            generator: torch.Generator | None = None,
            sway_coefficient: float | None = None,
            noise: torch.Tensor | None = None,
    ) -> LarryVocoderGeneration:
        frames = mels.shape[-1] * self.mel_upsample
        mask = make_non_pad_mask(self.spec_lengths(mel_lengths, frames), frames).to(mels.dtype)
        conditioning = self.upsample_mels(mels, frames)
        shape = (mels.shape[0], 2 * self.n_bins, frames)
        spec = self.sample(
            lambda x, t: self.backbone(x, t, conditioning, mask), shape, steps, self.device, generator, sway_coefficient,
            noise,
        )
        waveforms = self.codec.decode(spec, (frames - 1) * self.config.stft_hop_length)
        lengths = (mel_lengths * self.config.hop_length - self.config.stft_n_fft // 2).clamp(min=0)
        return LarryVocoderGeneration(waveforms=waveforms, waveform_lengths=lengths)


class LarryVocoderStream:
    def __init__(
            self,
            vocoder: LarryVocoder,
            steps: int,
            lookahead: int | None = None,
            left_context: int | None = None,
            crossfade: int | None = None,
            generator: torch.Generator | None = None,
            sway_coefficient: float | None = None,
    ) -> None:
        self.vocoder = vocoder
        self.steps = steps
        self.lookahead = vocoder.receptive_field if lookahead is None else lookahead
        self.left_context = vocoder.receptive_field if left_context is None else left_context
        self.hop_length = vocoder.config.hop_length
        self.crossfade = vocoder.config.stft_hop_length if crossfade is None else crossfade
        if not 0 < self.crossfade <= self.hop_length:
            raise ValueError(f"crossfade must be between 1 and {self.hop_length} samples")

        if self.lookahead * self.hop_length < self.crossfade + vocoder.config.stft_n_fft // 2:
            raise ValueError("lookahead must cover the crossfade plus half an STFT window")

        self.generator = generator
        self.sway_coefficient = sway_coefficient
        self.mels: torch.Tensor | None = None
        self.noise: torch.Tensor | None = None
        self.offset = 0
        self.emitted = 0
        self.tail: torch.Tensor | None = None

    @property
    def buffered(self) -> int:
        return self.offset + (0 if self.mels is None else self.mels.shape[-1])

    @torch.inference_mode()
    def push(self, mels: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        vocoder = self.vocoder
        if noise is None:
            shape = (mels.shape[0], 2 * vocoder.n_bins, mels.shape[-1] * vocoder.mel_upsample)
            noise = torch.randn(shape, device=mels.device, generator=self.generator)

        self.mels = mels if self.mels is None else torch.cat([self.mels, mels], dim=-1)
        self.noise = noise if self.noise is None else torch.cat([self.noise, noise], dim=-1)
        ready = self.buffered - self.lookahead
        if ready <= self.emitted:
            return mels.new_zeros(mels.shape[0], 0)

        return self.render(ready, final=False)

    @torch.inference_mode()
    def finish(self) -> torch.Tensor:
        if self.mels is None:
            raise ValueError("nothing was pushed")

        return self.render(self.buffered, final=True)

    def render(self, end: int, final: bool) -> torch.Tensor:
        assert self.mels is not None and self.noise is not None
        vocoder = self.vocoder
        upsample = vocoder.mel_upsample
        start = max(self.offset, self.emitted - self.left_context)
        window = self.mels[..., start - self.offset:]
        noise = self.noise[..., (start - self.offset) * upsample:]
        frames = window.shape[-1] * upsample
        mask = torch.ones(window.shape[0], frames, device=window.device, dtype=window.dtype)
        conditioning = vocoder.upsample_mels(window, frames)
        spec = vocoder.sample(
            lambda x, t: vocoder.backbone(x, t, conditioning, mask), tuple(noise.shape), self.steps, window.device,
            None, self.sway_coefficient, noise,
        )
        waveforms = vocoder.codec.decode(spec, (frames - 1) * vocoder.config.stft_hop_length)
        first = (self.emitted - start) * self.hop_length
        last = waveforms.shape[-1] if final else (end - start) * self.hop_length
        emitted = waveforms[..., first:last]
        if self.tail is not None:
            fade = self.tail.shape[-1]
            ramp = torch.linspace(0, 1, fade + 2, device=emitted.device)[1:-1]
            emitted = torch.cat([self.tail * (1 - ramp) + emitted[..., :fade] * ramp, emitted[..., fade:]], dim=-1)

        self.tail = None if final else waveforms[..., last:last + self.crossfade]
        self.emitted = end
        keep = max(self.offset, end - self.left_context)
        self.mels = self.mels[..., keep - self.offset:]
        self.noise = self.noise[..., (keep - self.offset) * upsample:]
        self.offset = keep
        return emitted
