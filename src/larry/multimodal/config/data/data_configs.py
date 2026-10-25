import dataclasses

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig
from larry.multimodal.config.model.config_builder import Task

SYNTHESIS_TASKS: set[Task] = {"text_only", "text_image", "text_voice"}


@dataclasses.dataclass(kw_only=True)
class MultimodalSourceConfig(DatasetSourceConfig):
    # each pretraining source holds one task, so the task also decides synthesis vs captioning
    task: Task
    text_column: str = "text"
    # exact text token counts for length bucketing, if precomputed; otherwise estimated from characters
    text_tokens_column: str | None = None
    voice_column: str = "unit_ids"
    # VAE latents stored flat, with their (C, H, W) shape alongside
    image_column: str = "latents"
    image_shape_column: str = "latent_shape"

    @property
    def is_modal_synthesis(self) -> bool:
        return self.task in SYNTHESIS_TASKS


@dataclasses.dataclass(kw_only=True)
class MultimodalMixtureConfig(DatasetMixtureConfig[MultimodalSourceConfig]):
    # padded stretched positions per batch, prompt and media together
    max_batch_tokens: int = 16384
    chars_per_token: float = 4.0
