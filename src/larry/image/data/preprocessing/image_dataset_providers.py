from typing import Mapping

from larry.common.data.cached_dataset_providers import MediaCachedDatasetProvider
from larry.image.config.preprocessing.image_dataset_configs import GoogleDOCCIMediaCachedDatasetConfig


class GoogleDOCCIMediaCachedDatasetProvider(MediaCachedDatasetProvider[GoogleDOCCIMediaCachedDatasetConfig]):
    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return {"train": str(self.download_extract_cache_dir / self.config.descriptions_file)}
