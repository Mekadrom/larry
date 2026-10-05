import logging
import os
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast, BinaryIO, ClassVar

from tqdm import tqdm

from larry.common.config.data.extractor_configs import ExtractorConfig, SingleFileExtractorConfig, \
    MultiFileExtractorConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Extractor[C: ExtractorConfig](Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[Extractor]]

    _config: C

    def __init__(self, config: C, extract_cache_abs_dir: Path) -> None:
        self._config = config
        self.extract_cache_dir = extract_cache_abs_dir

    @property
    def config(self) -> C:
        return self._config

    @abstractmethod
    def ensure_extracted(self) -> None:
        ...


class SingleFileExtractor(Extractor[SingleFileExtractorConfig]):
    def __init__(self, config: SingleFileExtractorConfig, extract_cache_abs_dir: Path):
        super().__init__(config, extract_cache_abs_dir)
        self.input_file_path = self.extract_cache_dir / config.input_file_name
        self.log = logging.getLogger(type(self).__name__)

    def ensure_extracted(self) -> None:
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
        total = os.path.getsize(file_path)
        with open(file_path, "rb") as stream:
            with tqdm.wrapattr(
                    stream, "read", total=total, unit="B", unit_scale=True, desc=str(file_path)) as pbar:
                with tarfile.open(fileobj=cast(BinaryIO, pbar), mode="r|*") as tar:
                    tar.extractall(output_dir)
        file_path.with_name(f"{file_path.name}.extracted").touch(exist_ok=False)


class MultiFileExtractor(Extractor[MultiFileExtractorConfig]):
    def __init__(self, config: MultiFileExtractorConfig, extract_cache_abs_dir: Path):
        super().__init__(config, extract_cache_abs_dir)

        self.delegates = [
            Extractor.REGISTRY.resolve(k)(c, self.extract_cache_dir)
            for k, c in config.delegate_configs
        ]

    def ensure_extracted(self) -> None:
        for delegate in self.delegates:
            delegate.ensure_extracted()
