import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.config.model.model_configs import LLMBlockModuleConfig
from larry.common.model.flow import TimestepEmbedding, ConditionalFlowMatching
from larry.common.model.llm import LarrySimpleEncoderBlock
from larry.common.model.norms import AdaLayerNorm
from larry.voice.config.model.voice_model_configs import SpeechDecoderModelConfig
from larry.voice.model.vocoder import ConditionalConvNeXtBlock
from larry.voice.model.voice_models import SpeechDecoderModelOutput, VoiceModel
from larry.voice.model.voice_modules import TTSLogMelFrontend
from larry.voice.utils import make_non_pad_mask


class SpeakerLatentEmbeddings(nn.Module):
    def __init__(self, n_speakers: int, d_model: int) -> None:
        super().__init__()

        self.mean_embedding = nn.Embedding(n_speakers, d_model)
        self.logvar_embedding = nn.Embedding(n_speakers, d_model)
        self.cfg_embedding = nn.Parameter(torch.zeros(d_model))

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


class UpsampleConv1dStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int, p_dropout: float) -> None:
        super().__init__()
        if scale % 2 != 0:
            raise ValueError(f"scale must be divisible by 2, got scale={scale}")

        d_inner = (out_channels + in_channels) // 2
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="linear"),
            nn.Conv1d(in_channels, d_inner, kernel_size=scale * 2, stride=scale, padding=scale // 2, groups=d_inner),
            nn.GroupNorm(d_inner),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(d_inner, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpeechCodesUnitEmbedding(nn.Module):
    def __init__(
            self,
            n_units: int,
            d_model: int,
            n_layers: int,
            encoder_dropout_p: float,
            upsample_dropout_p: float,
    ) -> None:
        super().__init__()

        self.unit_embedding = nn.Embedding(n_units, d_model)

        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(LLMBlockModuleConfig(d_model=d_model, mlp_dropout_p=encoder_dropout_p))
            for _ in range(n_layers)
        ])
        self.post_blocks_norm = nn.RMSNorm(d_model)

        self.upsample1 = UpsampleConv1dStack(
            in_channels=d_model,
            out_channels=d_model,
            scale=2,
            p_dropout=upsample_dropout_p
        )
        self.upsample2 = UpsampleConv1dStack(
            in_channels=d_model,
            out_channels=d_model,
            scale=2,
            p_dropout=upsample_dropout_p
        )

    def forward(self, unit_ids: torch.Tensor, unit_id_lengths: torch.Tensor) -> torch.Tensor:
        mask = make_non_pad_mask(unit_id_lengths).to(unit_ids.device)
        x = self.unit_embedding(unit_ids)
        for block in self.blocks:
            x = block(x, attention_mask=mask)

        x = self.post_blocks_norm(x)

        x = x + self.upsample1(x)
        return x + self.upsample2(x), mask


class SpeechDecoderVelocityNetwork(nn.Module):
    def __init__(self, n_mels: int, n_bins: int, n_speakers: int, d_model: int, d_inner: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Conv1d(n_mels + 2 * n_bins, d_model, kernel_size=7, padding=3)
        self.time = TimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([
            ConditionalConvNeXtBlock(d_model, d_inner, d_model, 1 / n_layers) for _ in range(n_layers)
        ])
        self.final_norm = AdaLayerNorm(d_model, d_model)
        self.out = nn.Linear(d_model, 2 * n_bins)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            cond: torch.Tensor,
            mels: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        cond = self.time(t) + cond
        x = self.embed(torch.cat([xt, mels], dim=1) * mask.unsqueeze(1))
        for block in self.blocks:
            x = block(x, cond, mask)

        return self.out(self.final_norm(x.transpose(1, 2), cond)).transpose(1, 2) * mask.unsqueeze(1)


class SpeechDecoderModel(VoiceModel[SpeechDecoderModelConfig], PyTorchModelHubMixin, ConditionalFlowMatching):
    def __init__(self, config: SpeechDecoderModelConfig) -> None:
        super().__init__(config)

        self.frontend = TTSLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
        self.speaker_embedding = SpeakerLatentEmbeddings(config.n_speakers, config.d_model)
        self.unit_encoder = SpeechCodesUnitEmbedding(
            n_units=config.n_units,
            d_model=config.d_model,
            n_layers=config.n_encoder_layers,
            encoder_dropout_p=config.encoder_dropout_p,
            upsample_dropout_p=config.upsample_dropout_p,
        )
        self.backbone = SpeechDecoderVelocityNetwork(
            n_mels=config.n_mels,
            n_bins=config.n_bins,
            n_speakers=config.n_speakers,
            d_model=config.d_model,
            d_inner=config.d_inner,
            n_layers=config.n_backbone_layers,
        )

    def forward(
            self,
            unit_ids: torch.Tensor,
            speaker_ids: torch.Tensor,
            waveforms: torch.Tensor,
            waveform_lengths: torch.Tensor
    ) -> SpeechDecoderModelOutput:
        speaker_embeddings, mu, logvar = self.speaker_latent_embeddings(speaker_ids)
        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        unit_embeddings, mask = self.unit_encoder(unit_ids)

        # timestep addition/flow matching bullshit
        ...

        kl_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kl = torch.sum(kl_element, dim=1)

        return SpeechDecoderModelOutput(
            logits=mels,
            speaker_mus=mu,
            speaker_logvars=logvar,
            kl=total_kl,
            speaker_embeddings=speaker_embeddings,
            target_mels=mels,
            target_mel_lengths=mel_lengths,
        )
