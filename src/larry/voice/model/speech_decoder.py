import dataclasses
from typing import cast

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.config.model.model_configs import LarryConditionalEncoderBlockModuleConfig, LLMBlockModuleConfig
from larry.common.model.flow import ConditionalFlowMatching, TimestepEmbedding
from larry.common.model.init import WeightInitializer
from larry.common.model.llm import LarryConditionalEncoderBlock, LarrySimpleEncoderBlock
from larry.common.model.masks import make_non_pad_mask
from larry.common.model.norms import AdaLayerNorm
from larry.common.training.training_model import GenerationOutput
from larry.voice.config.model.voice_model_configs import SpeechDecoderModelConfig
from larry.voice.model.prosody import F0RangePredictor, F0Statistics, HarmonicConditioning
from larry.voice.model.voice_models import SpeechDecoderModelOutput, VoiceModel
from larry.voice.model.voice_modules import TTSLogMelFrontend


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderGeneration(GenerationOutput):
    mels: torch.Tensor
    mel_lengths: torch.Tensor


class SpeakerLatentEmbeddings(nn.Module):
    def __init__(self, n_speakers: int, d_model: int) -> None:
        super().__init__()

        self.mean_embedding = nn.Embedding(n_speakers, d_model)
        self.logvar_embedding = nn.Embedding(n_speakers, d_model)

    def forward(self, speaker_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mean_embedding(speaker_ids)
        logvar = self.logvar_embedding(speaker_ids)
        # sampled while training; the mean at eval, so eval losses don't depend on the draw
        z = self._reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SpeechCodesUnitEmbedding(nn.Module):
    def __init__(self, config: SpeechDecoderModelConfig) -> None:
        super().__init__()

        self.upsample = config.unit_upsample
        self.unit_embedding = nn.Embedding(config.n_units, config.d_model)
        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(LLMBlockModuleConfig.from_child(config))
            for _ in range(config.n_encoder_layers)
        ])
        self.post_blocks_norm = nn.RMSNorm(config.d_model)
        self.upsampler = nn.ConvTranspose1d(
            config.d_model,
            config.d_model,
            kernel_size=2 * self.upsample,
            stride=self.upsample,
            padding=self.upsample // 2,
        )
        self.proj = nn.Linear(config.d_model, config.n_mels)

    def forward(self, unit_ids: torch.Tensor, unit_id_lengths: torch.Tensor) -> torch.Tensor:
        mask = make_non_pad_mask(unit_id_lengths, unit_ids.shape[1]).to(unit_ids.device)
        x = self.unit_embedding(unit_ids)
        for block in self.blocks:
            x = block(x, attention_mask=mask).logits

        x = self.post_blocks_norm(x) * mask.unsqueeze(-1).to(x.dtype)
        x = self.upsampler(x.transpose(1, 2))[..., :unit_ids.shape[1] * self.upsample]
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class SpeechDecoderVelocityNetwork(nn.Module):
    def __init__(self, config: SpeechDecoderModelConfig, prosody_channels: int = 0) -> None:
        super().__init__()
        in_channels = (3 if config.infill_enabled else 2) * config.n_mels + prosody_channels
        self.embed = nn.Conv1d(in_channels, config.d_model, kernel_size=7, padding=3)
        self.time = TimestepEmbedding(config.d_model)
        self.blocks = nn.ModuleList([
            LarryConditionalEncoderBlock(LarryConditionalEncoderBlockModuleConfig.from_child(config))
            for _ in range(config.n_backbone_layers)
        ])
        self.final_norm = AdaLayerNorm(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.n_mels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            speaker_cond: torch.Tensor,
            unit_cond: torch.Tensor,
            mask: torch.Tensor,
            context: torch.Tensor | None = None,
            prosody: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.time(t) + speaker_cond
        inputs = [xt, unit_cond] if context is None else [xt, unit_cond, context]
        if prosody is not None:
            inputs.append(prosody.to(xt.dtype))
        x = self.embed(torch.cat(inputs, dim=1) * mask.unsqueeze(1)).transpose(1, 2)
        for block in self.blocks:
            x = block(x, cond, mask).logits

        return self.out(self.final_norm(x, cond)).transpose(1, 2) * mask.unsqueeze(1)


class SpeechDecoderModel(VoiceModel[SpeechDecoderModelConfig], PyTorchModelHubMixin, ConditionalFlowMatching):
    def __init__(self, config: SpeechDecoderModelConfig) -> None:
        VoiceModel.__init__(self, config)
        ConditionalFlowMatching.__init__(self, config)

        self.frontend = TTSLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
        self.speaker_embedding = SpeakerLatentEmbeddings(config.n_speakers, config.speaker_dim)
        self.cfg_null_speaker_embedding = nn.Parameter(torch.zeros(config.speaker_dim))
        self.speaker_projection = nn.Linear(config.speaker_dim, config.d_model)
        self.unit_encoder = SpeechCodesUnitEmbedding(config)
        self.harmonics = None
        self.f0_range = None
        if config.prosody_conditioning:
            self.harmonics = HarmonicConditioning(
                config.harmonic_mode,
                cast(torch.Tensor, self.frontend.mel.mel_scale.fb),
                config.sample_rate,
                config.hop_length,
                config.n_harmonics,
                config.comb_sharpness,
                config.n_aperiodicity_bands if config.use_aperiodicity else None,
            )
            self.f0_range = F0RangePredictor(config.speaker_dim, config.f0_range_hidden_dim)

        self.backbone = SpeechDecoderVelocityNetwork(config, self.harmonics.channels if self.harmonics else 0)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        # DiT-style: xavier projections, small embeddings and timestep MLP, zeroed modulations and output
        initializer = WeightInitializer("xavier_uniform", 0.02)
        initializer.apply(self)
        for layer in self.backbone.time.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)

        initializer.scale_residuals(self.unit_encoder.blocks, len(self.unit_encoder.blocks))
        # posteriors start at unit variance, matching the KL prior
        nn.init.zeros_(self.speaker_embedding.logvar_embedding.weight)
        nn.init.zeros_(self.backbone.out.weight)
        nn.init.zeros_(self.backbone.out.bias)
        if self.f0_range is not None:
            self.f0_range.reset_parameters()

    @staticmethod
    def fit_frames(values: torch.Tensor, frames: int) -> torch.Tensor:
        if values.shape[1] >= frames:
            return values[:, :frames]

        padding = [0, 0] * (values.dim() - 2) + [0, frames - values.shape[1]]
        return F.pad(values, padding)

    def prosody(
            self,
            f0: torch.Tensor | None,
            aperiodicity: torch.Tensor | None,
            frame_mask: torch.Tensor,
            log_f0_mean: torch.Tensor | None = None,
            log_f0_spread: torch.Tensor | None = None,
            use_true_f0: bool = False,
    ) -> tuple[torch.Tensor | None, F0Statistics | None]:
        if self.harmonics is None:
            return None, None

        batch_size, frames = frame_mask.shape
        if f0 is None:
            return torch.zeros(batch_size, self.harmonics.channels, frames, device=frame_mask.device), None

        stats = F0Statistics.from_f0(self.fit_frames(f0.float(), frames), frame_mask)
        if use_true_f0:
            log_f0 = stats.log_f0
        else:
            assert log_f0_mean is not None and log_f0_spread is not None
            log_f0 = (log_f0_mean[:, None] + log_f0_spread[:, None] * stats.contour) * stats.voiced

        if aperiodicity is not None:
            aperiodicity = self.fit_frames(aperiodicity.float(), frames)

        features = self.harmonics(stats.contour, stats.voiced, log_f0, aperiodicity)
        return features * frame_mask.unsqueeze(1).to(features.dtype), stats

    def normalize(self, mels: torch.Tensor) -> torch.Tensor:
        return (mels - self.config.mel_mean) / self.config.mel_std

    def denormalize(self, mels: torch.Tensor) -> torch.Tensor:
        return mels * self.config.mel_std + self.config.mel_mean

    def infill_mask(self, lengths: torch.Tensor, frames: int) -> torch.Tensor:
        generator = self.noise_generator
        fraction = torch.empty(len(lengths), device=lengths.device).uniform_(
            self.config.infill_min_fraction, self.config.infill_max_fraction, generator=generator
        )
        span = (lengths.float() * fraction).long().clamp(min=1)
        offsets = torch.rand(len(lengths), device=lengths.device, generator=generator)
        start = (offsets * (lengths - span + 1).clamp(min=1).float()).long()
        positions = torch.arange(frames, device=lengths.device)[None]
        return (positions >= start[:, None]) & (positions < (start + span)[:, None])

    def forward(
            self,
            unit_ids: torch.Tensor,
            unit_id_lengths: torch.Tensor,
            speaker_ids: torch.Tensor,
            waveforms: torch.Tensor,
            waveform_lengths: torch.Tensor,
            f0: torch.Tensor | None = None,
            aperiodicity: torch.Tensor | None = None,
            use_true_f0: bool = False,
    ) -> SpeechDecoderModelOutput:
        speaker_embs, mu, logvar = self.speaker_embedding(speaker_ids)
        drop = torch.zeros(speaker_embs.shape[0], 1, dtype=torch.bool, device=speaker_embs.device)
        if self.training and self.config.cfg_dropout_rate > 0.0:
            drop = torch.rand(speaker_embs.shape[0], 1, device=speaker_embs.device) < self.config.cfg_dropout_rate
            speaker_embs = torch.where(drop, self.cfg_null_speaker_embedding, speaker_embs)

        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        unit_cond = self.unit_encoder(unit_ids, unit_id_lengths)

        frames = min(mels.shape[-1], unit_cond.shape[-1])
        mels, unit_cond = self.normalize(mels[..., :frames]), unit_cond[..., :frames]
        lengths = torch.minimum(mel_lengths, self.config.unit_upsample * unit_id_lengths).clamp(max=frames)
        frame_mask = make_non_pad_mask(lengths, frames).to(mels.dtype)

        context = None
        loss_mask = frame_mask
        if self.config.infill_enabled:
            hidden = self.infill_mask(lengths, frames)
            context = mels * (~hidden & ~drop).unsqueeze(1) * frame_mask.unsqueeze(1)
            loss_mask = frame_mask * hidden.to(frame_mask.dtype)

        f0_range = f0_range_targets = f0_range_mask = None
        prosody = None
        if self.f0_range is not None:
            mean, log_spread = self.f0_range(speaker_embs)
            prosody, stats = self.prosody(
                f0, aperiodicity, frame_mask > 0, mean.detach(), log_spread.detach().exp(), use_true_f0
            )
            if stats is not None and prosody is not None:
                f0_range = torch.stack([mean, log_spread], dim=1)
                f0_range_targets = torch.stack([stats.log_f0_mean, stats.log_f0_spread.log()], dim=1)
                f0_range_mask = stats.has_range
                if self.training and self.config.f0_dropout_rate > 0.0:
                    keep = torch.rand(prosody.shape[0], device=prosody.device) >= self.config.f0_dropout_rate
                    prosody = prosody * keep.view(-1, 1, 1).to(prosody.dtype)

        xt, t, target = self.path(mels)
        pred = self.backbone(xt, t, self.speaker_projection(speaker_embs), unit_cond, frame_mask, context, prosody)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return SpeechDecoderModelOutput(
            logits=pred,
            targets=target,
            mask=loss_mask,
            speaker_mus=mu,
            speaker_logvars=logvar,
            kl=kl,
            speaker_embeddings=speaker_embs,
            prior=unit_cond,
            prior_targets=mels,
            frame_mask=frame_mask,
            f0_range=f0_range,
            f0_range_targets=f0_range_targets,
            f0_range_mask=f0_range_mask,
        )

    def generate(
            self,
            unit_ids: torch.Tensor,
            unit_id_lengths: torch.Tensor,
            speaker_ids: torch.Tensor | None,
            steps: int,
            cfg_scale: float = 1.0,
            generator: torch.Generator | None = None,
            sway_coefficient: float | None = None,
            prompt_mels: torch.Tensor | None = None,
            prompt_mel_lengths: torch.Tensor | None = None,
            f0: torch.Tensor | None = None,
            aperiodicity: torch.Tensor | None = None,
            log_f0_mean: torch.Tensor | None = None,
            log_f0_spread: torch.Tensor | None = None,
    ) -> SpeechDecoderGeneration:
        if prompt_mels is not None and not self.config.infill_enabled:
            raise ValueError("prompt_mels requires a model trained with infill_enabled")

        unit_cond = self.unit_encoder(unit_ids, unit_id_lengths)
        frames = unit_cond.shape[-1]
        lengths = (self.config.unit_upsample * unit_id_lengths).clamp(max=frames)
        mask = make_non_pad_mask(lengths, frames).to(unit_cond.dtype)

        null_embedding = self.cfg_null_speaker_embedding.expand(unit_ids.shape[0], -1)
        speaker_vectors = null_embedding if speaker_ids is None else self.speaker_embedding.mean_embedding(speaker_ids)
        null_cond = self.speaker_projection(null_embedding)
        speaker_cond = self.speaker_projection(speaker_vectors)

        prosody = None
        if self.f0_range is not None:
            predicted_mean, predicted_log_spread = self.f0_range(speaker_vectors)
            mean = predicted_mean if log_f0_mean is None else log_f0_mean.float()
            spread = predicted_log_spread.exp() if log_f0_spread is None else log_f0_spread.float()
            prosody, _ = self.prosody(f0, aperiodicity, mask > 0, mean, spread)

        context = None
        prompt_mask = None
        if self.config.infill_enabled:
            context = torch.zeros(unit_ids.shape[0], self.config.n_mels, frames, device=unit_ids.device)
            if prompt_mels is not None:
                assert prompt_mel_lengths is not None
                width = min(prompt_mels.shape[-1], frames)
                prompt_mask = make_non_pad_mask(prompt_mel_lengths.clamp(max=width), frames).unsqueeze(1)
                context[..., :width] = self.normalize(prompt_mels[..., :width])
                context = context * prompt_mask

        if cfg_scale == 1.0:
            def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return self.backbone(x, t, speaker_cond, unit_cond, mask, context, prosody)
        else:
            both_speakers = torch.cat([speaker_cond, null_cond])
            both_units = torch.cat([unit_cond, unit_cond])
            both_masks = torch.cat([mask, mask])
            both_contexts = None if context is None else torch.cat([context, torch.zeros_like(context)])
            both_prosody = None if prosody is None else torch.cat([prosody, prosody])

            def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                conditional, unconditional = self.backbone(
                    torch.cat([x, x]), torch.cat([t, t]), both_speakers, both_units, both_masks, both_contexts,
                    both_prosody,
                ).chunk(2)
                return unconditional + cfg_scale * (conditional - unconditional)

        shape = (unit_ids.shape[0], self.config.n_mels, frames)
        mels = self.sample(velocity, shape, steps, unit_ids.device, generator, sway_coefficient)
        if prompt_mask is not None and context is not None:
            mels = torch.where(prompt_mask, context, mels)

        return SpeechDecoderGeneration(mels=self.denormalize(mels) * mask.unsqueeze(1), mel_lengths=lengths)
