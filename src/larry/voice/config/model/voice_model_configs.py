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
class ASRVoiceModelConfig(VoiceModelConfig):
    """Override for ``VoiceModelConfig`` that changes the n_fft for ASR models."""

    n_fft: int = 400


@dataclasses.dataclass(kw_only=True)
class SpeechTokenizerModelConfig(ASRVoiceModelConfig, LLMBlockModuleConfig):
    """Combines the ``ASRVoiceModelConfig`` and ``LLMBlockModuleConfig`` to configure a ``SpeechTokenizerModel``."""

    subsampler_stride: int = 2
    n_query_groups: int = 16

    d_model: int = 1024
    fsq_dims: int = 8
    fsq_levels: int = 3
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

