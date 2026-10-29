import dataclasses
import math
from typing import Literal

from larry.common.config.model.model_configs import (
    FlowMatchingConfig,
    LarryConditionalEncoderBlockModuleConfig,
    LLMBlockModuleConfig,
    ModelConfig,
    VocabularyConfig,
)
from larry.voice.config.preprocessing.voice_configs import MelsConfig


@dataclasses.dataclass(kw_only=True)
class VoiceModelConfig(ModelConfig, MelsConfig):
    """Superclass for voice model configs, setting global defaults."""


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderDecoderModelConfig(VoiceModelConfig, LLMBlockModuleConfig):
    fsq_levels: list[int] = dataclasses.field(default_factory=lambda: [3] * 8)

    @property
    def n_units(self) -> int:
        return math.prod(self.fsq_levels)


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderModelConfig(SpeechEncoderDecoderModelConfig, LLMBlockModuleConfig, VocabularyConfig):
    """Uses ``SpeechTokenizerModelConfig`` and ``LLMBlockModuleConfig`` to configure ``SpeechTokenizerEncoderModel``."""

    n_fft: int = 400
    n_mels: int = 100

    subsampler_stride: int = 2

    d_model: int = 1024

    n_encoder1_layers: int = 6
    n_premerge_layers: int = 3
    frame_merge: int = 2
    fsmn_kernel_size: int = 31
    merged_fsmn_kernel_size: int = 15
    n_encoder2_layers: int = 6
    ctc_upsample: int = 1

    vocab_size: int = 512

    attention_dropout_p: float = 0.0
    proj_dropout_p: float = 0.0
    mlp_dropout_p: float = 0.0

    n_spec_freq_masks: int = 2
    spec_freq_width: int = 27
    n_spec_time_masks: int = 10
    spec_time_fraction: float = 0.05

    enable_fsq_ctc_head: bool = False

    weight_init: Literal["pytorch", "scaled_normal"] = "pytorch"
    init_std: float = 0.02


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderModelConfig(
    SpeechEncoderDecoderModelConfig, LarryConditionalEncoderBlockModuleConfig, FlowMatchingConfig
):
    """Uses ``SpeechTokenizerModelConfig`` and ``LLMBlockModuleConfig`` to configure ``SpeechDecoderModel``."""

    n_speakers: int

    d_model: int = 1024
    speaker_dim: int = 192
    n_encoder_layers: int = 4
    n_backbone_layers: int = 4
    hop_length: int = 320
    unit_upsample: int = 4

    is_causal: bool = False
    attention_dropout_p: float = 0.0
    proj_dropout_p: float = 0.0
    mlp_dropout_p: float = 0.0

    mel_mean: float = 0.0
    mel_std: float = 1.0

    cfg_dropout_rate: float = 0.1

    infill_enabled: bool = False
    infill_min_fraction: float = 0.7
    infill_max_fraction: float = 1.0


@dataclasses.dataclass(kw_only=True)
class LarryVocoderConfig(VoiceModelConfig, FlowMatchingConfig):
    """Configures ``LarryVocoder`` models."""

    hop_length: int = 320
    stft_n_fft: int = 640
    stft_hop_length: int = 160

    d_model: int = 512
    d_inner: int = 1536
    n_layers: int = 6

    stft_alpha: float = 0.5
    stft_scale: float = 1.0
