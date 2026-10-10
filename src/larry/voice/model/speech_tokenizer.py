import dataclasses

import torch
import torch.nn.functional as F
import torchaudio.transforms
from torch import nn

from larry.common.config.model.model_configs import FSMNAttentionConfig
from larry.common.model.attention import FSMNAttention
from larry.common.model.llm import LarrySimpleEncoderBlock
from larry.common.model.model import LarrySpeechTokenizerModelOutput
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.model.fsq import FSQ
from larry.voice.model.voice_models import VoiceModel
from larry.voice.utils import make_non_pad_mask


class SpeechTokenizerEncoderBlock(nn.Module):
    def __init__(self, config: SpeechTokenizerModelConfig) -> None:
        super().__init__()
        self.attn = FSMNAttention(FSMNAttentionConfig.from_child(config))
        self.attn_norm = nn.LayerNorm(config.d_model)

        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )
        self.mlp_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.attn_norm(x), attention_mask=pad_mask)
        x = x + attended
        return x + self.mlp(self.mlp_norm(x))


class SpeechEncoderPreBottleneck(VoiceModel[SpeechTokenizerModelConfig]):
    """Log-mels at 100 Hz -> two strided convs -> FSMN attention blocks -> [B, T, D] at 25 Hz."""

    def __init__(self, config: SpeechTokenizerModelConfig) -> None:
        super().__init__(config)

        self.kernel_size = 3
        self.padding = 1
        self.stride1 = config.subsampler_stride
        self.stride2 = 2

        self.conv1 = nn.Conv1d(
            config.n_mels,
            config.d_model,
            kernel_size=self.kernel_size,
            stride=self.stride1,
            padding=self.padding,
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=self.kernel_size,
            stride=self.stride2,
            padding=self.padding,
        )

        self.blocks = nn.ModuleList([
            SpeechTokenizerEncoderBlock(config)
            for _ in range(config.n_encoder1_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def _conv_lengths(self, lengths: torch.Tensor, stride: int) -> torch.Tensor:
        return (lengths + 2 * self.padding - self.kernel_size) // stride + 1

    def forward(self, mels: torch.Tensor, mel_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # mels: [B, n_mels, T_mel], mel_lengths: [B]
        mask = make_non_pad_mask(mel_lengths, mels.shape[-1])
        x = F.gelu(self.conv1(mels * mask.unsqueeze(1)))

        lengths = self._conv_lengths(mel_lengths, self.stride1)
        mask = make_non_pad_mask(lengths, x.shape[-1])
        x = F.gelu(self.conv2(x * mask.unsqueeze(1)))

        lengths = self._conv_lengths(lengths, self.stride2)
        mask = make_non_pad_mask(lengths, x.shape[-1])

        x = x.transpose(1, 2)  # [B, T, D]
        for block in self.blocks:
            x = block(x, mask)

        # zero padded frames so nothing downstream (FSQ, encoder2) sees leftover values there
        x = self.final_norm(x) * mask.unsqueeze(-1).to(x.dtype)
        return x, lengths


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int, n_fft: int = 400, hop_length: int = 160) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            mels = self.mel(waveforms.float()).clamp(min=1e-10).log()  # [B, n_mels, T]
        mel_lengths = lengths // self.hop_length + 1
        return mels, mel_lengths


class SpeechTokenizerModel(VoiceModel[SpeechTokenizerModelConfig]):
    def __init__(self, config: SpeechTokenizerModelConfig) -> None:
        super().__init__(config)

        encoder_config = dataclasses.replace(config, is_causal=False)

        self.encoder1 = SpeechEncoderPreBottleneck(encoder_config)
        self.fsq = FSQ(config.d_model, dims=config.fsq_dims, levels=config.fsq_levels)

        self.encoder2 = nn.ModuleList([
            LarrySimpleEncoderBlock(encoder_config)
            for _ in range(config.n_encoder2_layers)
        ])
        self.encoder2_norm = nn.LayerNorm(config.d_model)

        self.ctc_head = nn.Linear(config.d_model, config.text_vocab_size + 1)

        self.frontend = LogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> LarrySpeechTokenizerModelOutput:
        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        encoded, frame_lengths = self.encoder1(mels, mel_lengths)
        frame_mask = make_non_pad_mask(frame_lengths, encoded.shape[1])

        quantized, token_ids = self.fsq(encoded)

        hidden = quantized
        for block in self.encoder2:
            hidden = block(hidden, attention_mask=frame_mask)
        hidden = self.encoder2_norm(hidden)

        ctc_log_probs = F.log_softmax(self.ctc_head(hidden).float(), dim=-1)

        return LarrySpeechTokenizerModelOutput(
            logits=ctc_log_probs,
            ctc_log_probs=ctc_log_probs,
            frame_lengths=frame_lengths,
            token_ids=token_ids,
        )

    @torch.inference_mode()
    def tokenize(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        encoded, frame_lengths = self.encoder1(mels, mel_lengths)
        _, token_ids = self.fsq(encoded)
        return token_ids, frame_lengths
