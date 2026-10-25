import dataclasses
import math

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.config.model.model_configs import FSMNAttentionConfig
from larry.common.model.attention import FSMNAttention
from larry.common.model.init import WeightInitializer
from larry.common.model.llm import LarrySimpleEncoderBlock
from larry.common.model.masks import make_non_pad_mask
from larry.voice.config.model.voice_model_configs import SpeechEncoderModelConfig
from larry.voice.model.fsq import FSQ
from larry.voice.model.voice_models import SpeechEncoderModelOutput, VoiceModel
from larry.voice.model.voice_modules import ASRLogMelFrontend


class SpeechEncoderEncoderBlock(nn.Module):
    """A conformer-style encoder block specific to the encoder modules in the ``SpeechTokenizerModel``.

    Uses ``FSMNAttention`` instead of ``LarryAttention`` to make use of a local memory branch Conv1d head to add to the
    residual.
    """

    def __init__(self, config: SpeechEncoderModelConfig) -> None:
        """Initializes modules."""
        super().__init__()
        self.attn = FSMNAttention(FSMNAttentionConfig.from_child(config))
        self.attn_norm = nn.RMSNorm(config.d_model)

        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )
        self.mlp_norm = nn.RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Simple encoder block forward with norms and a residual."""
        x = x + self.attn(self.attn_norm(x), attention_mask=pad_mask).logits
        return x + self.mlp(self.mlp_norm(x))


class SpeechEncoderPreBottleneck(VoiceModel[SpeechEncoderModelConfig], register=False):
    """Log-mels at 100 Hz -> two strided convs -> FSMN attention blocks -> [B, T, D] at 25 Hz.

    Consists of a convolutional subsampling frontend followed by a set of ``SpeechTokenizerEncoderBlock`` instances to
    feed the input of the ``FSQ`` bottleneck.
    """

    def __init__(self, config: SpeechEncoderModelConfig) -> None:
        """Initializes modules."""
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
            SpeechEncoderEncoderBlock(config)
            for _ in range(config.n_encoder1_layers)
        ])
        self.final_norm = nn.RMSNorm(config.d_model)

    @property
    def frame_seconds(self) -> float:
        """Helper for calculating the fractional seconds in each frame."""
        return self.config.hop_length * self.stride1 * self.stride2 / self.config.sample_rate

    def _conv_lengths(self, lengths: torch.Tensor, stride: int) -> torch.Tensor:
        """A normalized length for each example that the strided convs will actually see."""
        return (lengths + 2 * self.padding - self.kernel_size) // stride + 1

    def forward(self, mels: torch.Tensor, mel_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Passes log-mels through two strided subsampling Conv1d modules and then a transformer encoder stack."""
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


class SpeechEncoderModel(VoiceModel[SpeechEncoderModelConfig], PyTorchModelHubMixin):
    """Takes waveforms and performs CTC prediction after a unit ID bottleneck."""

    def __init__(self, config: SpeechEncoderModelConfig) -> None:
        """Initializes modules.

        Includes the ``LogMelFrontend``, ``SpeechEncoderPreBottleneck``, ``FSQ``, encoder2 transformer encoder block
        stack, and CTC prediction head.
        """
        super().__init__(config)

        encoder_config = dataclasses.replace(config, is_causal=False)

        self.frontend = ASRLogMelFrontend(config.sample_rate, config.n_mels, config.n_fft, config.hop_length)

        self.encoder1 = SpeechEncoderPreBottleneck(encoder_config)
        self.fsq = FSQ(config.d_model, dims=config.fsq_dims, levels=config.fsq_levels)

        if config.enable_fsq_ctc_head:
            self.fsq_ctc_head = nn.Linear(config.d_model, config.vocab_size + 1)
        else:
            self.fsq_ctc_head = None

        self.encoder2 = nn.ModuleList([
            LarrySimpleEncoderBlock(encoder_config)
            for _ in range(config.n_encoder2_layers)
        ])
        self.encoder2_norm = nn.RMSNorm(config.d_model)

        self.ctc_head = nn.Linear(config.d_model, config.vocab_size + 1)

        if config.weight_init == "scaled_normal":
            self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        initializer = WeightInitializer("normal", self.config.init_std)
        initializer.apply(self)

        # the subsampler feeds GELUs straight from log-mels; He init keeps the activation scale through both convs
        for conv in (self.encoder1.conv1, self.encoder1.conv2):
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

        # the FSQ bottleneck separates two residual streams
        initializer.scale_residuals(self.encoder1.blocks, len(self.encoder1.blocks))
        initializer.scale_residuals(self.encoder2, len(self.encoder2))

        # unit-variance pre-tanh activations, so FSQ starts out using every level instead of rounding most dims to 0
        nn.init.normal_(self.fsq.down.weight, std=1 / math.sqrt(self.config.d_model))

        # CTC heads start uniform over tokens and blank
        for head in (self.ctc_head, self.fsq_ctc_head):
            if head is not None:
                nn.init.zeros_(head.weight)

    @staticmethod
    def _spec_augment(
            mels: torch.Tensor,
            lengths: torch.Tensor,
            freq_masks: int,
            freq_width: int,
            time_masks: int,
            time_fraction: float
    ) -> torch.Tensor:
        """LD-style SpecAugment on [B, n_mels, T] log-mels; masked cells take each clip's mean."""
        batch_size, freq_bins, seq_len = mels.shape
        device = mels.device
        valid = (torch.arange(seq_len, device=device).view(1, 1, seq_len) < lengths.view(batch_size, 1, 1))
        fill = ((mels * valid).sum(dim=(1, 2)) / (lengths.clamp(min=1) * freq_bins)).view(batch_size, 1, 1)

        f = torch.arange(freq_bins, device=device).view(1, 1, freq_bins)
        f_width = torch.randint(0, freq_width + 1, (batch_size, freq_masks, 1), device=device)
        f_start = (torch.rand(batch_size, freq_masks, 1, device=device) * (freq_bins - f_width)).long()
        f_mask = ((f >= f_start) & (f < f_start + f_width)).any(dim=1)  # [B, F]

        t = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        max_width = (lengths.float() * time_fraction).view(batch_size, 1, 1)
        t_width = (torch.rand(batch_size, time_masks, 1, device=device) * max_width).long()
        t_start = (torch.rand(batch_size, time_masks, 1, device=device) * (lengths.view(batch_size, 1, 1) - t_width))
        t_start = t_start.long()
        t_mask = ((t >= t_start) & (t < t_start + t_width)).any(dim=1)  # [B, T]

        masked = f_mask.unsqueeze(2) | t_mask.unsqueeze(1)
        return torch.where(masked, fill, mels)

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> SpeechEncoderModelOutput:
        """Given waveforms, produces a model output contained both the bottleneck token IDs and CTC ASR predictions."""
        ctc_log_probs = {}

        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        if self.training and (self.config.n_spec_time_masks > 0 or self.config.n_spec_freq_masks > 0):
            mels = self._spec_augment(
                mels,
                mel_lengths,
                self.config.n_spec_freq_masks,
                self.config.spec_freq_width,
                self.config.n_spec_time_masks,
                self.config.spec_time_fraction
            )

        encoded, frame_lengths = self.encoder1(mels, mel_lengths)
        frame_mask = make_non_pad_mask(frame_lengths, encoded.shape[1])

        quantized, unit_ids = self.fsq(encoded)

        if self.fsq_ctc_head is not None:
            ctc_log_probs["fsq"] = F.log_softmax(self.fsq_ctc_head(quantized).float(), dim=-1)

        hidden = quantized
        for block in self.encoder2:
            hidden = block(hidden, attention_mask=frame_mask).logits

        hidden = self.encoder2_norm(hidden)

        canonical_ctc_log_probs = F.log_softmax(self.ctc_head(hidden).float(), dim=-1)
        ctc_log_probs["encoder2_norm"] = canonical_ctc_log_probs
        return SpeechEncoderModelOutput(
            logits=canonical_ctc_log_probs,
            ctc_log_probs=ctc_log_probs,
            frame_lengths=frame_lengths,
            unit_ids=unit_ids,
            mels=mels,
            mel_lengths=mel_lengths,
        )

    @torch.inference_mode()
    def tokenize(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The "generate" function for a model like this that takes waveforms and produces the bottleneck unit IDs."""
        mels, mel_lengths = self.frontend(waveforms, waveform_lengths)
        encoded, frame_lengths = self.encoder1(mels, mel_lengths)
        _, unit_ids = self.fsq(encoded)
        return unit_ids, frame_lengths
