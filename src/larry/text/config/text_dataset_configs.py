import dataclasses

from larry.common.config.dataset_config import DatasetConfig


@dataclasses.dataclass
class TextDatasetConfig(DatasetConfig):
    """Super class for all DTOs related to text dataset preprocessing."""


@dataclasses.dataclass
class MonologyPileUncopyrightedDatasetConfig(TextDatasetConfig):
    """https://huggingface.co/datasets/monology/pile-uncopyrighted"""

    path = "monology/pile-uncopyrighted"
    provenance_columns = [
        "meta"
    ]