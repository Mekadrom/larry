from typing import Sequence, Mapping

from larry.common.data.cached_dataset_providers import MediaCachedDatasetProvider
from larry.image.config.image_dataset_configs import GoogleDOCCIMediaCachedDatasetConfig


class GoogleDOCCIMediaCachedDatasetProvider(MediaCachedDatasetProvider[GoogleDOCCIMediaCachedDatasetConfig]):
    @property
    def load_input_data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        return [str((self.dir / GoogleDOCCIMediaCachedDatasetConfig.DESCRIPTIONS_FILE).expanduser().resolve())]
