import torch
from torch import nn

from larry.common.model.model import LarrySpeechDecoderModelOutput
from larry.voice.config.model.voice_model_configs import SpeechDecoderModelConfig
from larry.voice.model.voice_models import VoiceModel
from larry.voice.model.voice_modules import ASRLogMelFrontend


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


class SpeakerCodesUnitEmbedding(nn.Module):
    def __init__(self, n_units: int, d_model: int, n_mels: int) -> None:
        super().__init__()

        self.unit_embedding = nn.Embedding(n_units, d_model)

        ...

    def forward(self, unit_ids: torch.Tensor) -> torch.Tensor:
        ...


class SpeakerGuidedConditionalFlowMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class SpeechDecoderModel(VoiceModel[SpeechDecoderModelConfig]):
    def __init__(self, config: SpeechDecoderModelConfig) -> None:
        super().__init__(config)

        self.speaker_latent_embeddings = SpeakerLatentEmbeddings(config.n_speakers, config.d_model)
        self.frontend = ASRLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)
        self.unit_encoder = SpeakerCodesUnitEmbedding(config.n_units, config.d_model, config.n_mels)

        self.timestep_embedding = nn.Embedding(10, config.d_model)

    def forward(
            self,
            unit_ids: torch.Tensor,
            speaker_ids: torch.Tensor,
            waveforms: torch.Tensor,
            waveform_lengths: torch.Tensor
    ) -> LarrySpeechDecoderModelOutput:
        speaker_embeddings, mu, logvar = self.speaker_latent_embeddings(speaker_ids)
        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        unit_embeddings = self.unit_encoder(unit_ids)

        # timestep addition/flow matching bullshit
        ...

        return LarrySpeechDecoderModelOutput(
            logits=mels,
            speaker_mu=mu,
            speaker_logvar=logvar,
            target_mels=mels,
            target_mel_lengths=mel_lengths,
        )
