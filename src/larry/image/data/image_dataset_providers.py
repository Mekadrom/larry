import logging
from pathlib import Path

from larry.common.data.dataset_providers import MediaCachedDatasetProvider
from larry.image.config.image_dataset_configs import GoogleDOCCIMediaCachedDatasetConfig

log = logging.getLogger(__name__)


class GoogleDOCCIMediaCachedDatasetProvider(MediaCachedDatasetProvider):
    config: GoogleDOCCIMediaCachedDatasetConfig

    def __init__(self, config: GoogleDOCCIMediaCachedDatasetConfig) -> None:
        super().__init__(config=config)
        self.config = config  # type override

    @property
    def split_file(self) -> Path:
        return (self.dir / GoogleDOCCIMediaCachedDatasetConfig.DESCRIPTIONS_FILE).expanduser().resolve()
