import dataclasses

from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig


@dataclasses.dataclass(kw_only=True)
class TextDatasetConfig(DatasetConfig):
    ...
