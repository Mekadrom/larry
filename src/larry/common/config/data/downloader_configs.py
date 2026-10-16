import dataclasses
from collections.abc import Sequence
from typing import ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class DownloaderConfig(Registrable, root=True):
    """Superclass for configs for all ``Downloader`` implementations."""

    REGISTRY: ClassVar[TypeRegistry[DownloaderConfig]]

    busy_wait: float = 0.0
    max_retries: int = 0


@dataclasses.dataclass(kw_only=True)
class SingleFileDownloaderConfig(DownloaderConfig):
    """Superclass for configs for all ``SingleFileDownloader`` implementations."""

    download_base_url: str
    output_file_name: str
    download_path: str | None = None
    download_timeout: int = 10

    validate_content_contains: str | None = None


@dataclasses.dataclass(kw_only=True)
class SingleFileBackupDownloaderConfig(SingleFileDownloaderConfig):
    """Placeholder subclass for ``SingleFileBackupDownloader`` in case it ever needs additional specific config."""

    ...


@dataclasses.dataclass(kw_only=True)
class MultiFileDownloaderConfig(DownloaderConfig):
    """Config class for ``MultiFileDownloader``.

    Allows for the configuration of multiple ordered delegate ``SingleFileDownloaderConfigs``.
    """

    configs: Sequence[tuple[str, SingleFileDownloaderConfig]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class RangedIndexMultiFileDownloaderConfig(MultiFileDownloaderConfig):
    """Config class for ``RangedIndexMultiFileDownloader``."""

    def configs_for(self, range_value: int) -> Sequence[tuple[str, DownloaderConfig]]:
        """Allows for overriding to return a ranged set of ``DownloaderConfig`` instances."""
        ...

    range_desc: str
