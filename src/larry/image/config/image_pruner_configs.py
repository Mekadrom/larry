import dataclasses

from larry.common.config.pruner_configs import PrunerConfig


@dataclasses.dataclass(kw_only=True)
class ImagePrunerConfig(PrunerConfig):
    ...
