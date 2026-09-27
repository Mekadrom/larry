import logging
from pathlib import Path

from larry.common.data.dataset_providers import MediaCachedDatasetProvider
from larry.voice.config.voice_dataset_configs import CommonVoiceMediaCachedDatasetConfig

log = logging.getLogger(__name__)


class CommonVoiceMediaCachedDatasetProvider(MediaCachedDatasetProvider):
    config: CommonVoiceMediaCachedDatasetConfig

    def __init__(self, config: CommonVoiceMediaCachedDatasetConfig) -> None:
        super().__init__(config=config)
        self.config = config  # type override

    @property
    def media_dir_path(self) -> Path:
        return super().media_dir_path / "audio"

    @property
    def split_file(self) -> Path:
        return (self.dir / f"{self.split}.tsv").expanduser().resolve()
