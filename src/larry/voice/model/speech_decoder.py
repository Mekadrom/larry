import dataclasses

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.config.model.model_configs import LarryConditionalEncoderBlockModuleConfig, LLMBlockModuleConfig
from larry.common.model.flow import ConditionalFlowMatching, TimestepEmbedding
from larry.common.model.llm import LarryConditionalEncoderBlock, LarrySimpleEncoderBlock
from larry.common.model.norms import AdaLayerNorm
from larry.common.training.training_model import GenerationOutput
from larry.voice.config.model.voice_model_configs import SpeechDecoderModelConfig
from larry.voice.model.voice_models import SpeechDecoderModelOutput, VoiceModel
from larry.voice.model.voice_modules import TTSLogMelFrontend
from larry.voice.utils import make_non_pad_mask


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
        z = self._reparameterize(mu, logvar)
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
        self.proj = nn.Linear(config.d_model, config.n_mels)

    def forward(self, unit_ids: torch.Tensor, unit_id_lengths: torch.Tensor) -> torch.Tensor:
        mask = make_non_pad_mask(unit_id_lengths, unit_ids.shape[1]).to(unit_ids.device)
        x = self.unit_embedding(unit_ids)
        for block in self.blocks:
            x = block(x, attention_mask=mask)

        x = self.post_blocks_norm(x)
        x = x.repeat_interleave(self.upsample, dim=1)
        return self.proj(x).transpose(1, 2)


class SpeechDecoderVelocityNetwork(nn.Module):
    def __init__(self, config: SpeechDecoderModelConfig) -> None:
        super().__init__()
        self.embed = nn.Conv1d(2 * config.n_mels, config.d_model, kernel_size=7, padding=3)
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
    ) -> torch.Tensor:
        cond = self.time(t) + speaker_cond
        x = self.embed(torch.cat([xt, unit_cond], dim=1) * mask.unsqueeze(1)).transpose(1, 2)
        for block in self.blocks:
            x = block(x, cond, mask)

        return self.out(self.final_norm(x, cond)).transpose(1, 2) * mask.unsqueeze(1)


class SpeechDecoderModel(VoiceModel[SpeechDecoderModelConfig], PyTorchModelHubMixin, ConditionalFlowMatching):
    def __init__(self, config: SpeechDecoderModelConfig) -> None:
        VoiceModel.__init__(self, config)
        ConditionalFlowMatching.__init__(self, config.sigma_min)

        self.frontend = TTSLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
        self.speaker_embedding = SpeakerLatentEmbeddings(config.n_speakers, config.speaker_dim)
        self.cfg_null_speaker_embedding = nn.Parameter(torch.zeros(config.speaker_dim))
        self.speaker_projection = nn.Linear(config.speaker_dim, config.d_model)
        self.unit_encoder = SpeechCodesUnitEmbedding(config)
        self.backbone = SpeechDecoderVelocityNetwork(config)

    def normalize(self, mels: torch.Tensor) -> torch.Tensor:
        return (mels - self.config.mel_mean) / self.config.mel_std

    def denormalize(self, mels: torch.Tensor) -> torch.Tensor:
        return mels * self.config.mel_std + self.config.mel_mean

    def forward(
            self,
            unit_ids: torch.Tensor,
            unit_id_lengths: torch.Tensor,
            speaker_ids: torch.Tensor,
            waveforms: torch.Tensor,
            waveform_lengths: torch.Tensor,
    ) -> SpeechDecoderModelOutput:
        speaker_embs, mu, logvar = self.speaker_embedding(speaker_ids)
        if self.training and self.config.cfg_dropout_rate > 0.0:
            drop = torch.rand(speaker_embs.shape[0], 1, device=speaker_embs.device) < self.config.cfg_dropout_rate
            speaker_embs = torch.where(drop, self.cfg_null_speaker_embedding, speaker_embs)

        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        unit_cond = self.unit_encoder(unit_ids, unit_id_lengths)

        frames = min(mels.shape[-1], unit_cond.shape[-1])
        mels, unit_cond = self.normalize(mels[..., :frames]), unit_cond[..., :frames]
        lengths = torch.minimum(mel_lengths, self.config.unit_upsample * unit_id_lengths)
        mask = make_non_pad_mask(lengths, frames).to(mels.dtype)

        xt, t, target = self.path(mels)
        pred = self.backbone(xt, t, self.speaker_projection(speaker_embs), unit_cond, mask)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return SpeechDecoderModelOutput(
            logits=pred,
            targets=target,
            mask=mask,
            speaker_mus=mu,
            speaker_logvars=logvar,
            kl=kl,
            speaker_embeddings=speaker_embs,
        )

    def generate(
            self,
            unit_ids: torch.Tensor,
            unit_id_lengths: torch.Tensor,
            speaker_ids: torch.Tensor,
            steps: int,
            cfg_scale: float = 1.0,
            generator: torch.Generator | None = None,
    ) -> SpeechDecoderGeneration:
        unit_cond = self.unit_encoder(unit_ids, unit_id_lengths)
        lengths = self.config.unit_upsample * unit_id_lengths
        mask = make_non_pad_mask(lengths, unit_cond.shape[-1]).to(unit_cond.dtype)
        speaker_cond = self.speaker_projection(self.speaker_embedding.mean_embedding(speaker_ids))

        if cfg_scale == 1.0:
            def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return self.backbone(x, t, speaker_cond, unit_cond, mask)
        else:
            null_cond = self.speaker_projection(self.cfg_null_speaker_embedding).expand_as(speaker_cond)
            both_speakers = torch.cat([speaker_cond, null_cond])
            both_units = torch.cat([unit_cond, unit_cond])
            both_masks = torch.cat([mask, mask])

            def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                conditional, unconditional = self.backbone(
                    torch.cat([x, x]), torch.cat([t, t]), both_speakers, both_units, both_masks
                ).chunk(2)
                return unconditional + cfg_scale * (conditional - unconditional)

        shape = (unit_ids.shape[0], self.config.n_mels, unit_cond.shape[-1])
        mels = self.sample(velocity, shape, steps, unit_ids.device, generator)
        return SpeechDecoderGeneration(mels=self.denormalize(mels) * mask.unsqueeze(1), mel_lengths=lengths)
