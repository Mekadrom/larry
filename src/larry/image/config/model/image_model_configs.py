import dataclasses

from larry.common.config.model.model_configs import ModelConfig


@dataclasses.dataclass(kw_only=True)
class ImageModelConfig(ModelConfig):
    ...
