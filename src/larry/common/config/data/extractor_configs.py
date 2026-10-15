import dataclasses
import logging
from collections.abc import Sequence


@dataclasses.dataclass(kw_only=True)
class ExtractorConfig:
    """Superclass for all configs for ``Extractor`` implementations."""
    ...


@dataclasses.dataclass(kw_only=True)
class SingleFileExtractorConfig(ExtractorConfig):
    """Config class for all ``SingleFileExtractor`` implementations."""
    input_file_name: str
    archive_root: str

    def __post_init__(self) -> None:
        """Required in order to default the value of ``archive_root``.

         Defaults to a supposedly sensible value consisting of the target archive name minus its file extension.
         """
        self.archive_root = self.archive_root or self._strip_archive_suffix(self.input_file_name)

    @staticmethod
    def _strip_archive_suffix(file_name: str) -> str:
        """Helper static function to remove archive file extensions from the target extractable file name."""
        for suffix in (".tar.gz", ".tgz", ".tar.xz", ".tar", ".zip"):
            if file_name.endswith(suffix):
                return file_name.removesuffix(suffix)

        logging.info(
            f"Did not remove suffix from file_name={file_name}, "
            f"considering adding this suffix: {file_name.split('.')[-1]}"
        )
        return file_name
