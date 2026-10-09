import dataclasses
import logging
from collections.abc import Sequence


def _strip_archive_suffix(file_name: str) -> str:
    for suffix in (".tar.gz", ".tgz", ".tar.xz", ".tar", ".zip"):
        if file_name.endswith(suffix):
            return file_name.removesuffix(suffix)
    logging.info(
        f"Did not remove suffix from file_name={file_name}, "
        f"considering adding this suffix: {file_name.split('.')[-1]}"
    )
    return file_name


@dataclasses.dataclass(kw_only=True)
class ExtractorConfig:
    ...


@dataclasses.dataclass(kw_only=True)
class SingleFileExtractorConfig(ExtractorConfig):
    input_file_name: str
    archive_root: str

    def __post_init__(self) -> None:
        self.archive_root = self.archive_root or _strip_archive_suffix(self.input_file_name)


@dataclasses.dataclass(kw_only=True)
class MultiFileExtractorConfig(ExtractorConfig):
    delegate_configs: Sequence[tuple[str, SingleFileExtractorConfig]] = dataclasses.field(default_factory=list)
