import dataclasses

from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.multimodal.config.criterion.criterion_configs import LarryMultimodalCriterionConfig
from larry.multimodal.config.data.data_configs import MultimodalMixtureConfig


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
