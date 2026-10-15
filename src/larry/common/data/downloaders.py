import logging
import os
import time
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote, urljoin

import requests
from tqdm import tqdm

from larry.common.config.data.downloader_configs import DownloaderConfig, SingleFileDownloaderConfig, \
    MultiFileDownloaderConfig, RangedIndexMultiFileDownloaderConfig, SingleFileBackupDownloaderConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


# TODO: merge with UrlMapper somehow or have them backed by the same downloading helper to deduplicate code


class Downloader[C: DownloaderConfig = DownloaderConfig](Registrable, ABC, root=True):
    """Superclass for all downloaders."""
    REGISTRY: ClassVar[TypeRegistry[Downloader]]

    _config: C

    def __init__(self, config: C, download_cache_abs_dir: Path) -> None:
        """Sets up the config."""
        self.log = logging.getLogger(type(self).__name__)

        self._config = config
        self.download_cache_abs_dir = download_cache_abs_dir

    @property
    def config(self) -> C:
        """Immutable type-hinted getter for the config instance."""
        return self._config

    @abstractmethod
    def ensure_downloaded(self) -> bool:
        """Returns True for downloaded, False for the downloaded file existing already at the target file path.

        Throws if the download is unsuccessful.
        """
        ...


UA = "larry-scotus-dataset-provider/1.0 (dataset research; contact via github.com/Mekadrom)"

_SAFE = "/%!$&'()*+,;=:@-._~"


class SingleFileDownloader[C: SingleFileDownloaderConfig = SingleFileDownloaderConfig](Downloader[C]):
    """Simple implementation of ``Downloader`` that downloads a single file (usually an archive or manifest)."""

    def __init__(self, config: C, download_cache_abs_dir: Path) -> None:
        """Sets up the config and state of this downloader instance."""
        super().__init__(config, download_cache_abs_dir)

        self.output_file_path = download_cache_abs_dir / config.output_file_name
        self.download_path = config.download_path or config.output_file_name
        self.download_timeout = config.download_timeout

        self.download_base_url = config.download_base_url
        if self.download_base_url is not None and not self.download_base_url.endswith("/"):
            self.download_base_url = self.download_base_url + "/"

        self.url = self._build_url(self.download_base_url, self.download_path)

    def _build_url(self, base_url: str, path: str) -> str:
        """Encodes a URL using the provided ``base_url`` and ``path``."""
        return self._encode_url(urljoin(base_url, quote(path)))

    @staticmethod
    def _encode_url(url: str) -> str:
        """Encodes a URL string, returning the encoded version."""
        p = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit([
            p.scheme,
            p.netloc,
            urllib.parse.quote(p.path, safe=_SAFE),
            urllib.parse.quote(p.query, safe=_SAFE + "?&="),
            p.fragment
        ])

    def ensure_downloaded(self) -> bool:
        """Ensures that the file this downloader instance is responsible for is downloaded.

        Checks for a cached file using the hash of the URL, returning early and with False if the cached file already
        exists. If it doesn't, downloads and returns True if the download was successful. Raises is downloading fails
        too many times. Supports busy wait sleep timers for throttling scraping.
        """
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
                content = self._download_file(self.url, temp_file_path, return_contents=True)
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
        """Actually performs the download, using tqdm to track progress against the total expected content length."""
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
        """Validates the content that was downloaded with a simple string content contains check if configured."""
        if self.config.validate_content_contains is not None:
            if self.config.validate_content_contains.encode("utf-8") in content:
                # downloaded and validated
                return True
        else:
            # downloaded and no validation
            return True
        return False


class SingleFileBackupDownloader(SingleFileDownloader[SingleFileBackupDownloaderConfig], ABC):
    """A specialized subclass of ``SingleFileDownloader`` that has a way to generated backup URLs configured."""

    def __init__(self, config: SingleFileBackupDownloaderConfig, download_cache_abs_dir: Path) -> None:
        """Sets up the config for this instance of ``SingleFileBackupDownloader``."""
        super().__init__(config, download_cache_abs_dir)
        self.base_output_file_path = self.output_file_path
        self.backup_download_paths = self.make_backup_download_paths(self.download_path)

    def ensure_downloaded(self) -> bool:
        """Ensures that the file this ``SingleFileBackupDownloader`` is responsible for is downloaded.

        Checks in the base output file path first, and if that file exists it returns early with False. If that file
        does not exist, continues on to download from the configured primary source. Any time a download source fails
        its allotment of retries, it tries to generate a new backup URL to download from using the configured
        backup_download_paths and the same download_base_url.
        """
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
                    self.url = self._build_url(self.download_base_url, self.backup_download_paths.pop())
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
        """Override to produce all possible backup download paths for this ``SingleFileBackupDownloader`` instance."""
        ...


class MultiFileDownloader[C: MultiFileDownloaderConfig = MultiFileDownloaderConfig](Downloader[C]):
    """An implementation of ``Downloader`` that delegates to multiple other ``Downloader`` implementations."""

    def __init__(self, config: C, download_cache_abs_dir: Path) -> None:
        """Instantiates the delegates on this instance's instantiation using the provided configs."""
        super().__init__(config, download_cache_abs_dir)

        self.delegates = [
            Downloader.REGISTRY.resolve(k)(c, self.download_cache_abs_dir)
            for k, c in self._delegate_configs()
        ]

    def ensure_downloaded(self) -> bool:
        """Just delegates to each delegate's ``ensure_downloaded``.

        Returns False if any delegate was required to perform a real download. Also performs busy wait throttling.
        """
        read_from_cache = True
        for i, delegate in enumerate(self.delegates):
            if delegate.ensure_downloaded():
                if i < len(self.delegates) - 1:
                    time.sleep(self.config.busy_wait)
            else:
                read_from_cache = False

        return read_from_cache

    def _delegate_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        """Returns the configured delegate configs."""
        return self.config.configs


class RangedIndexMultiFileDownloader[C: RangedIndexMultiFileDownloaderConfig = RangedIndexMultiFileDownloaderConfig](
    MultiFileDownloader[C]
):
    """An implementation of ``MultiFileDownloader`` that produces a set of delegates according to a range definition."""

    def _delegate_configs(self) -> Sequence[tuple[str, DownloaderConfig]]:
        """Creates delegates according to a range definition."""
        return [
            config
            for value in self._get_range_values(self.config.range_desc)
            for config in self.config.configs_for(value)
        ]

    @staticmethod
    def _get_range_values(range_desc: str) -> Sequence[int]:
        """Reads in a range config like "2010-2025" or "2010,2011,2012,2014,2015" to return a parsed list of ints."""
        if "-" in range_desc:
            n_min, n_max = (int(s.strip()) for s in range_desc.strip().split("-"))
            return range(n_min, n_max + 1)
        elif "," in range_desc:
            return [int(s.strip()) for s in range_desc.strip().split(",")]
        
        return [int(range_desc.strip())]
