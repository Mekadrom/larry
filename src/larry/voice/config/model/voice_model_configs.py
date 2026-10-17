import dataclasses

from larry.common.config.model.model_configs import LLMBlockModuleConfig, ModelConfig
from larry.voice.config.preprocessing.voice_configs import SampleRateConfig


@dataclasses.dataclass(kw_only=True)
class VoiceModelConfig(ModelConfig, SampleRateConfig):
    """Superclass for voice model configs, setting global defaults."""

    n_fft: int = 1024
    n_mels: int = 100
    hop_length: int = 160


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderDecoderModelConfig(VoiceModelConfig, LLMBlockModuleConfig):
    fsq_dims: int = 8
    fsq_levels: int = 3

    @property
    def n_units(self) -> int:
        return self.fsq_levels ** self.fsq_dims


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderModelConfig(SpeechEncoderDecoderModelConfig, LLMBlockModuleConfig):
    """Uses ``SpeechTokenizerModelConfig`` and ``LLMBlockModuleConfig`` to configure ``SpeechTokenizerEncoderModel``."""

    n_fft: int = 400

    subsampler_stride: int = 2
    n_query_groups: int = 16

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


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderModelConfig(SpeechEncoderDecoderModelConfig, LLMBlockModuleConfig):
    """Uses ``SpeechTokenizerModelConfig`` and ``LLMBlockModuleConfig`` to configure ``SpeechDecoderModel``."""
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 480
