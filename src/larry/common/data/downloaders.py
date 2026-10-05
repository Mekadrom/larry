import logging
import os
import time
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Sequence
from urllib.parse import quote, urljoin

import requests
from tqdm import tqdm

from larry.common.config.data.downloader_configs import DownloaderConfig, SingleFileDownloaderConfig, \
    MultiFileDownloaderConfig, RangedIndexMultiFileDownloaderConfig, SingleFileBackupDownloaderConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Downloader[C: DownloaderConfig = DownloaderConfig](Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[Downloader]]

    _config: C

    def __init__(self, config: C, download_cache_abs_dir: Path) -> None:
        self._config = config
        self.download_cache_abs_dir = download_cache_abs_dir

        self.log = logging.getLogger(type(self).__name__)

    @property
    def config(self) -> C:
        return self._config

    @abstractmethod
    def ensure_downloaded(self) -> bool:
        """Returns True for downloaded, False for the downloaded file existing already at the target file path.

        Throws if the download is unsuccessful.
        """
        ...


UA = "larry-scotus-dataset-provider/1.0 (dataset research; contact via github.com/Mekadrom)"

_SAFE = "/%!$&'()*+,;=:@-._~"


def encode_url(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((
        p.scheme, p.netloc,
        urllib.parse.quote(p.path, safe=_SAFE),
        urllib.parse.quote(p.query, safe=_SAFE + "?&="),
        p.fragment)
    )


class SingleFileDownloader[C: SingleFileDownloaderConfig = SingleFileDownloaderConfig](Downloader[C]):
    def __init__(self, config: C, download_cache_abs_dir: Path) -> None:
        super().__init__(config, download_cache_abs_dir)

        self.output_file_path = download_cache_abs_dir / config.output_file_name
        self.download_path = config.download_path or config.output_file_name
        self.download_timeout = config.download_timeout

        self.download_base_url = config.download_base_url
        if self.download_base_url is not None and not self.download_base_url.endswith("/"):
            self.download_base_url = self.download_base_url + "/"

        self.url = self.build_url(self.download_base_url, self.download_path)

    def build_url(self, base_url: str, path: str) -> str:
        return encode_url(urljoin(base_url, quote(path)))

    def ensure_downloaded(self) -> bool:
        if os.path.exists(self.output_file_path):
            self.log.info(f"Already downloaded: output_file_path={self.output_file_path}")
            return False

        os.makedirs(self.output_file_path.parent, exist_ok=True)

        attempt = 0
        content = None
        last: Exception | None = None
        while attempt <= self.config.max_retries:
            attempt += 1
            temp_file_path = self.output_file_path.with_name(f"{self.output_file_path.name}.part")
            try:
                content = self._download_file(self.url, self.output_file_path, return_contents=True)
            except Exception as e:
                self.log.critical(f"Error downloading file_name={self.output_file_path} from download_path={self.url}")
                last = e
                content = None

            if content is not None and self._validate(content):
                temp_file_path.move(self.output_file_path)
                break

            time.sleep(self.config.busy_wait)

        if content is None:
            if last is None:
                raise ValueError(f"Could not download file after {self.config.max_retries + 1} tries: {self.url}")
            raise ValueError(f"Could not download file after {self.config.max_retries + 1} tries: {self.url}") from last
        return True

    def _download_file(self, url: str, file_path: Path, return_contents: bool = False) -> bytes | None:
        with requests.get(url, stream=True, timeout=self.download_timeout, headers={"User-Agent": UA}) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(file_path, 'wb') as stream:
                with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=str(file_path.name)) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        pbar.update(len(chunk))
        if return_contents:
            return file_path.read_bytes()
        return None

    def _validate(self, content: bytes) -> bool:
        if self.config.validate_content_contains is not None:
            if self.config.validate_content_contains in content:
                # downloaded and validated
                return True
        else:
            # downloaded and no validation
            return True
        return False


class SingleFileBackupDownloader(SingleFileDownloader[SingleFileBackupDownloaderConfig], ABC):
    def __init__(self, config: SingleFileBackupDownloaderConfig, download_cache_abs_dir: Path) -> None:
        super().__init__(config, download_cache_abs_dir)
        self.base_output_file_path = self.output_file_path
        self.backup_download_paths = self.make_backup_download_paths(self.download_path)

    def ensure_downloaded(self) -> bool:
        if os.path.exists(self.base_output_file_path):
            return False

        success = False
        exhausted = False
        last = None
        while not success and not exhausted:
            try:
                super().ensure_downloaded()
                success = True
            except ValueError as e:
                last = e
                self.log.info(f"Failed to download {self.url}, retrying...")
                time.sleep(self.config.busy_wait)
                if len(self.backup_download_paths) > 0:
                    self.url = self.build_url(self.download_base_url, self.backup_download_paths.pop())
                else:
                    exhausted = True

        if exhausted:
            if last is None:
                raise ValueError(
                    f"Could not download file after {self.config.max_retries} retries: {self.url}; "
                    f"exhausted backups: {self.backup_download_paths}"
                )
            raise ValueError(
                f"Could not download file after {self.config.max_retries} retries: {self.url}; "
                f"exhausted backups: {self.backup_download_paths}"
            ) from last
        return True

    @abstractmethod
    def make_backup_download_paths(self, default_url: str) -> list[str]:
        ...


class MultiFileDownloader[C: MultiFileDownloaderConfig = MultiFileDownloaderConfig](Downloader[C]):
    def __init__(self, config: C, download_cache_abs_dir: Path) -> None:
        super().__init__(config, download_cache_abs_dir)

        self.delegates = [
            Downloader.REGISTRY.resolve(k)(c, self.download_cache_abs_dir)
            for k, c in self._delegate_configs()
        ]

    def ensure_downloaded(self) -> bool:
        read_from_cache = True
        for i, delegate in enumerate(self.delegates):
            if delegate.ensure_downloaded():
                read_from_cache = False
                if i < len(self.delegates) - 1:
                    time.sleep(self.config.busy_wait)
        return read_from_cache

    def _delegate_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return self.config.configs

class RangedIndexMultiFileDownloader[C: RangedIndexMultiFileDownloaderConfig = RangedIndexMultiFileDownloaderConfig](
    MultiFileDownloader[C]
):
    def _delegate_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        return [
            config
            for value in self._get_range_values(self.config.range_desc)
            for config in self.config.configs_for(value)
        ]

    @staticmethod
    def _get_range_values(range_desc: str) -> Sequence[int]:
        if "-" in range_desc:
            n_min, n_max = (int(s.strip()) for s in range_desc.strip().split("-"))
            return range(n_min, n_max + 1)
        elif "," in range_desc:
            return [int(s.strip()) for s in range_desc.strip().split(",")]
        return [int(range_desc.strip())]
