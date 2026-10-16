import logging
import os
import tarfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, ClassVar, cast

from tqdm import tqdm

from larry.common.config.data.extractor_configs import ExtractorConfig, SingleFileExtractorConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Extractor[C: ExtractorConfig](Registrable, ABC, root=True):
    """Superclass for extractors."""

    REGISTRY: ClassVar[TypeRegistry[Extractor]]

    _config: C

    def __init__(self, config: C, extract_cache_abs_dir: Path) -> None:
        """Sets up this instance's config."""
        self.log = logging.getLogger(type(self).__name__)

        self._config = config
        self.extract_cache_dir = extract_cache_abs_dir

    @property
    def config(self) -> C:
        """Immutable type-hint config getter."""
        return self._config

    @abstractmethod
    def ensure_extracted(self) -> None:
        """Entrypoint for all extractors to ensure that a specified archive is extracted."""
        ...


class SingleFileExtractor(Extractor[SingleFileExtractorConfig]):
    """Extractor implementation responsible for extracting a single archive file."""

    def __init__(self, config: SingleFileExtractorConfig, extract_cache_abs_dir: Path) -> None:
        """Sets up this instance's config."""
        super().__init__(config, extract_cache_abs_dir)
        self.input_file_path = self.extract_cache_dir / config.input_file_name

    def ensure_extracted(self) -> None:
        """Ensures that the specified archive has been extracted.

        First checks to see if the input file path exists as a primary form of validation. Then, checks to see if a
        special ".extracted" marker file for the specified archive file exists. If it does, then this archive has been
        extracted already and it returns early. If it does not exist, moves on to extracting.
        """
        if not os.path.exists(self.input_file_path):
            raise ValueError(f"input_file_path does not exist to extract: {self.input_file_path}")

        if os.path.exists(self.input_file_path.with_name(f"{self.input_file_path.name}.extracted")):
            self.log.info(f"Already extracted: input_file_path={self.input_file_path}")
            return

        try:
            self._extract_file(self.extract_cache_dir, self.input_file_path)
        except Exception:
            self.log.critical(f"Error extracting tar at {self.input_file_path}")
            raise

    @staticmethod
    def _extract_file(output_dir: Path, file_path: Path) -> None:
        """Performs the extraction of supported file types; mostly tarballs or zips. Supports tqdm progress bars."""
        name = file_path.name
        if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")):
            total = os.path.getsize(file_path)
            with open(file_path, "rb") as stream, tqdm.wrapattr(
                    stream,
                    "read",
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=str(file_path)
            ) as pbar, tarfile.open(fileobj=cast(BinaryIO, pbar), mode="r|*") as tar:
                tar.extractall(output_dir)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(file_path) as archive:
                members = archive.infolist()
                total = sum([m.file_size for m in members])
                with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=str(file_path)
                ) as pbar:
                    for member in members:
                        archive.extract(member, output_dir)
                        pbar.update(member.file_size)
        else:
            raise ValueError(f"unsupported archive type: {name}")
        file_path.with_name(f"{file_path.name}.extracted").touch(exist_ok=False)
