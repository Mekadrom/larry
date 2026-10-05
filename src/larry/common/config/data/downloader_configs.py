import dataclasses
from typing import ClassVar, Sequence

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class DownloaderConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[DownloaderConfig]]

    busy_wait: float = 0.0
    max_retries: int = 0


@dataclasses.dataclass(kw_only=True)
class SingleFileDownloaderConfig(DownloaderConfig):
    download_base_url: str
    output_file_name: str
    download_path: str | None = None
    download_timeout: int = 10

    validate_content_contains: str | None = None


@dataclasses.dataclass(kw_only=True)
class SingleFileBackupDownloaderConfig(SingleFileDownloaderConfig):
    ...


@dataclasses.dataclass(kw_only=True)
class MultiFileDownloaderConfig(DownloaderConfig):
    configs: Sequence[tuple[str, SingleFileDownloaderConfig]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class RangedIndexMultiFileDownloaderConfig(MultiFileDownloaderConfig):
    def configs_for(self, range_value: int) -> Sequence[tuple[str, DownloaderConfig]]:
        ...

    range_desc: str
