import dataclasses

from larry.common.config.model.model_configs import ModelConfig, LLMBlockModuleConfig


@dataclasses.dataclass(kw_only=True)
class VoiceModelConfig(ModelConfig):
    sample_rate: int = 16000
    n_fft: int = 1024
    n_mels: int = 80
    hop_length: int = 256


@dataclasses.dataclass(kw_only=True)
class ASRVoiceModelConfig(VoiceModelConfig):
    n_fft: int = 400
    n_mels: int = 100
    hop_length: int = 160


@dataclasses.dataclass(kw_only=True)
class SpeechTokenizerModelConfig(ASRVoiceModelConfig, LLMBlockModuleConfig):
    subsampler_stride: int = 2

    d_model: int = 1024
    fsq_dims: int = 8
    fsq_levels: int = 3
    n_encoder1_layers: int = 6
    n_encoder2_layers: int = 6

    text_vocab_size: int = 512
