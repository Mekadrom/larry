import dataclasses

from larry.common.config.config import Config
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.multimodal.config.criterion.criterion_configs import LarryMultimodalCriterionConfig
from larry.multimodal.config.data.data_configs import MultimodalMixtureConfig


@dataclasses.dataclass(kw_only=True)
class MultimodalVizConfig(Config):
    examples_per_source: int = 2
    max_new_tokens: int = 48
    voice_temperature: float = 0.8
    # generated units run this many times the reference's groups, plus slack, before being cut off
    voice_length_budget: float = 1.5
    image_steps: int = 16
    image_cfg_scale: float = 2.0
    seed: int = 0
    # speech decoder and vocoder checkpoints turn units into audio; without them voice is judged on units alone
    speech_decoder_model_path: str | None = None
    vocoder_model_path: str | None = None
    vocoder_steps: int = 16


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalTrainerConfig(LarryTrainerConfig[MultimodalMixtureConfig]):
    processor_model_path: str
    # the encoder whose FSQ produced the datasets' unit ids; its config sets the voice vocab
    speech_encoder_model_path: str | None = None
    # channels of the VAE latents the datasets store
    image_latent_channels: int | None = None
    criterion: LarryMultimodalCriterionConfig = dataclasses.field(default_factory=LarryMultimodalCriterionConfig)
    # each checkpoint also gets a from_pretrained-ready copy of the model with its processor
    export_on_save: bool = True
    viz: MultimodalVizConfig = dataclasses.field(default_factory=MultimodalVizConfig)
