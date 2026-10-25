import dataclasses

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig
from larry.multimodal.config.model.config_builder import Task

SYNTHESIS_TASKS: set[Task] = {"text_only", "text_image", "text_voice"}


@dataclasses.dataclass(kw_only=True)
class MultimodalSourceConfig(DatasetSourceConfig):
    # each pretraining source holds one task, so the task also decides synthesis vs captioning
    task: Task
    text_column: str = "text"
    voice_column: str = "unit_ids"
    image_column: str = "latents"

    @property
    def is_modal_synthesis(self) -> bool:
        return self.task in SYNTHESIS_TASKS


@dataclasses.dataclass(kw_only=True)
class MultimodalMixtureConfig(DatasetMixtureConfig[MultimodalSourceConfig]):
    ...
