import dataclasses

from larry.common.config.data.preprocessing.pruner_configs import PrunerConfig


@dataclasses.dataclass(kw_only=True)
class TextPrunerConfig(PrunerConfig):
    ...
