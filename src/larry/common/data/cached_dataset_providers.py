import glob
import os.path
import tarfile
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Sequence, Mapping, Any
from urllib.parse import urljoin, quote

import datasets
import requests
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets.table import embed_table_storage
from tqdm import tqdm

from larry.common.config.dataset_config import CachedDatasetConfig, MediaCachedDatasetConfig
from larry.common.data.dataset_providers import HuggingFaceDatasetProvider
from larry.parquet_utils import save_parquet_shards


class CachedDatasetProvider(HuggingFaceDatasetProvider, ABC):
    config: CachedDatasetConfig

    def __init__(self, config: CachedDatasetConfig) -> None:
        super().__init__(config)
        self.config = config  # type override

        self.dir = Path(self.config.download_extract_cache_dir).expanduser().resolve()
        self.parquet_dir = self.dir / "parquets"
        self.default_download_url = self.config.default_download_url

        if self.default_download_url is not None and not self.default_download_url.endswith("/"):
            self.default_download_url = self.default_download_url + "/"

    @property
    def path(self) -> str | None:
        return "parquet"

    @property
    def data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        return glob.glob(str(self.parquet_dir / f"{self.split}-*.parquet"))

    @property
    @abstractmethod
    def split_file(self) -> Path:
        ...

    def load_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        os.makedirs(self.dir, exist_ok=True)

        self.log.info("Downloading...")
        self.ensure_downloaded()
        self.log.info("Done downloading.")

        self.log.info("Extracting...")
        self.ensure_extracted()
        self.log.info("Done extracting.")

        self.log.info("Caching as parquets...")
        self._ensure_parquets()
        self.log.info(f"Done caching as parquets at {self.parquet_dir}")

        kwargs = self.load_dataset_kwargs

        kwargs.pop("name", None)
        kwargs.pop("data_dir", None)

        if self.streaming:
            return datasets.load_dataset(**kwargs, streaming=True)
        return datasets.load_dataset(**kwargs, streaming=False)

    def ensure_downloaded(self) -> None:
        for file_name, download_extract_config in self.config.download_extract_configs.items():
            output_file_path = self.dir / file_name
            download_path = download_extract_config.download_path

            if download_path is None:
                if not os.path.exists(output_file_path):
                    raise ValueError(f"Could not download and was not already downloaded: {output_file_path}")
                continue

            if os.path.exists(output_file_path):
                self.log.info(f"Already downloaded: output_file_path={output_file_path}")
                continue

            url = urljoin(
                download_extract_config.override_download_base_url or self.default_download_url,
                quote(download_path)
            )

            try:
                self._download_file(url, output_file_path)
            except Exception:
                self.log.critical(f"Error downloading file_name={output_file_path} from download_path={url}")
                raise

    def _download_file(self, url: str, file_path: Path) -> None:
        temp_file_path = file_path.with_name(f"{file_path.name}.part")
        with requests.get(url, stream=True, timeout=self.config.download_timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(temp_file_path, 'wb') as stream:
                with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=str(file_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        pbar.update(len(chunk))
        temp_file_path.move(file_path)

    def ensure_extracted(self) -> None:
        for file_name, download_extract_config in self.config.download_extract_configs.items():
            try:
                input_file_path = self.dir / file_name
                extract_dir = download_extract_config.archive_root

                if not os.path.exists(input_file_path):
                    raise ValueError(f"input_file_path does not exist to extract: {input_file_path}")

                if extract_dir is None:
                    self.log.info(f"Extraction not supported for input_file_path={input_file_path}, skipping")
                    continue

                if os.path.exists(input_file_path.with_name(f"{input_file_path.name}.extracted")):
                    self.log.info(f"Already extracted: input_file_path={input_file_path}")
                    continue

                self._extract_file(self.dir, input_file_path)
            except Exception:
                self.log.critical(f"Error extracting tar at {file_name}")
                raise

    @staticmethod
    def _extract_file(output_dir: Path, file_path: Path) -> None:
        total = os.path.getsize(file_path)
        with open(file_path, "rb") as stream:
            with tqdm.wrapattr(
                    stream, "read", total=total, unit="B", unit_scale=True, desc=str(file_path)) as pbar:
                # noinspection PyTypeChecker
                with tarfile.open(fileobj=pbar, mode="r|*") as tar:
                    tar.extractall(str(output_dir))
        file_path.with_name(f"{file_path.name}.extracted").touch(exist_ok=False)

    def _ensure_parquets(self) -> None:
        os.makedirs(self.parquet_dir, exist_ok=True)

        output_parquet_file = self.parquet_dir / f"{self.split}.parquets.complete"
        if os.path.exists(output_parquet_file):
            self.log.info(f"parquets already exist: output_parquet_file={output_parquet_file}")
            return

        for file in self.data_files:
            Path(file).unlink()

        kw = self.config.download_extract_transform_kwargs
        kw.update(self.load_dataset_kwargs)

        kw.pop("path", None)
        kw.pop("name", None)
        kw.pop("data_files", None)
        kw.pop("data_dir", None)
        kw.pop("split", None)
        kw.pop("streaming", None)

        ds = datasets.load_dataset(
            path=self.config.download_extract_file_type,
            data_files=str(self.split_file),
            split="train",
            **kw,
        )

        if self.config.select_columns_from_extracted is not None:
            ds = ds.select_columns(self.config.select_columns_from_extracted)

        ds = ds.map(
            self.transform_batch,
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
        )
        ds = self.cast_dirs(ds)

        ds = ds.with_format("arrow").map(embed_table_storage, batched=True).with_format(None)

        save_parquet_shards(ds, self.parquet_dir, self.config.cached_parquet_size_mb, self.split)

        leftover = [f["filename"] for f in ds.cache_files]
        ds.cleanup_cache_files()
        for f in leftover:
            Path(f).unlink(missing_ok=True)

    @abstractmethod
    def cast_dirs(self, dataset: Dataset) -> Dataset:
        ...

    @abstractmethod
    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...


class MediaCachedDatasetProvider(CachedDatasetProvider, ABC):
    config: MediaCachedDatasetConfig

    def __init__(self, config: MediaCachedDatasetConfig) -> None:
        super().__init__(config=config)
        self.config = config

    @property
    def media_dir_path(self) -> Path:
        return self.dir / self.config.download_extract_configs[self.config.media_file_name].archive_root

    def cast_dirs(self, dataset: Dataset) -> Dataset:
        return dataset.cast_column(self.config.media_column, self.config.media_type)

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            self.config.media_column: [str(self.media_dir_path / p) for p in batch[self.config.media_path_column]]
        }
