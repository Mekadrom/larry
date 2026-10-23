import dataclasses

from larry.common.config.model.model_configs import (
    FlowMatchingConfig,
    LarryConditionalEncoderBlockModuleConfig,
    LLMBlockModuleConfig,
    ModelConfig,
    TextVocabularyConfig,
)
from larry.voice.config.preprocessing.voice_configs import SampleRateConfig


@dataclasses.dataclass(kw_only=True)
class VoiceModelConfig(ModelConfig, SampleRateConfig):
    """Superclass for voice model configs, setting global defaults."""

    n_fft: int = 640
    n_mels: int = 80
    hop_length: int = 160


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderDecoderModelConfig(VoiceModelConfig, LLMBlockModuleConfig):
    fsq_dims: int = 8
    fsq_levels: int = 3

    @property
    def n_units(self) -> int:
        return self.fsq_levels ** self.fsq_dims


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderModelConfig(SpeechEncoderDecoderModelConfig, LLMBlockModuleConfig, TextVocabularyConfig):
    """Uses ``SpeechTokenizerModelConfig`` and ``LLMBlockModuleConfig`` to configure ``SpeechTokenizerEncoderModel``."""

    n_fft: int = 400
    n_mels: int = 100

    subsampler_stride: int = 2

    d_model: int = 1024

    n_encoder1_layers: int = 6
    n_encoder2_layers: int = 6

    text_vocab_size: int = 512

    attention_dropout_p: float = 0.0
    proj_dropout_p: float = 0.0
    mlp_dropout_p: float = 0.0

    n_spec_freq_masks: int = 2
    spec_freq_width: int = 27
    n_spec_time_masks: int = 10
    spec_time_fraction: float = 0.05

    enable_fsq_ctc_head: bool = False


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
    d_model: int = 512
    d_inner: int = 1536
    n_layers: int = 6

    stft_alpha: float = 0.5
    stft_scale: float = 1.0
