import csv
import glob
import logging
import os.path
import tarfile
from abc import abstractmethod, ABC
from functools import partial
from pathlib import Path
from typing import Sequence, Mapping, Any
from urllib.parse import urljoin, quote

import datasets
import requests
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset, Features, Value
from datasets.table import embed_table_storage
from tqdm import tqdm

from larry.common.config.dataset_config import CachedDatasetConfig, MediaCachedDatasetConfig, DownloaderConfig, \
    ExtractorConfig
from larry.common.data.dataset_providers import HuggingFaceDatasetProvider
from larry.parquet_utils import save_parquet_shards


class Downloader:
    def __init__(self, config: DownloaderConfig, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.download_path = config.download_path
        self.output_file_path = self.cache_dir / config.output_file_name
        self.download_timeout = config.download_timeout

        self.download_url = config.download_url
        if self.download_url is not None and not self.download_url.endswith("/"):
            self.download_url = self.download_url + "/"

        self.log = logging.getLogger(__name__)

    def ensure_downloaded(self) -> None:
        if os.path.exists(self.output_file_path):
            self.log.info(f"Already downloaded: output_file_path={self.output_file_path}")
            return

        url = urljoin(self.download_url, quote(self.download_path))

        try:
            self._download_file(url)
        except Exception:
            self.log.critical(f"Error downloading file_name={self.output_file_path} from download_path={url}")
            raise

    def _download_file(self, url: str) -> None:
        file_path = self.output_file_path
        temp_file_path = file_path.with_name(f"{file_path.name}.part")
        with requests.get(url, stream=True, timeout=self.download_timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(temp_file_path, 'wb') as stream:
                with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=str(file_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        pbar.update(len(chunk))
        temp_file_path.move(file_path)


class Extractor:
    def __init__(self, config: ExtractorConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.input_file_name = config.input_file_name
        self.log = logging.getLogger(__name__)

    def ensure_extracted(self) -> None:
        input_file_path = self.cache_dir / self.input_file_name

        if not os.path.exists(input_file_path):
            raise ValueError(f"input_file_path does not exist to extract: {input_file_path}")

        if os.path.exists(input_file_path.with_name(f"{input_file_path.name}.extracted")):
            self.log.info(f"Already extracted: input_file_path={input_file_path}")
            return

        try:
            self._extract_file(self.cache_dir, input_file_path)
        except Exception:
            self.log.critical(f"Error extracting tar at {input_file_path}")
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


class CachedDatasetProvider[C=CachedDatasetConfig](HuggingFaceDatasetProvider[C], ABC):
    def __init__(self, config: CachedDatasetConfig) -> None:
        super().__init__(config)

        self.dir = Path(self.config.download_extract_cache_dir).expanduser().resolve()
        self.parquet_dir = self.dir / "parquets"

        self.downloaders = [Downloader(c, self.dir) for c in self.config.downloader_configs]
        self.extractors = [Extractor(c, self.dir) for c in self.config.extractor_configs]

    @property
    def path(self) -> str | None:
        return "parquet"

    @property
    def data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        return {
            split: glob.glob(str(self.parquet_dir / f"{split}/{split}-*.parquet"))
            for split in os.listdir(self.parquet_dir)
        }

    @property
    @abstractmethod
    def load_input_data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        ...

    def load_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        os.makedirs(self.dir, exist_ok=True)

        self.log.info("Downloading...")
        for downloader in self.downloaders:
            downloader.ensure_downloaded()
        self.log.info("Done downloading.")

        self.log.info("Extracting...")
        for extractor in self.extractors:
            extractor.ensure_extracted()
        self.log.info("Done extracting.")

        self.load_secondary_data()

        self.log.info("Caching as parquets...")
        try:
            self._ensure_parquets()
        except Exception:
            logging.critical(f"Exception occurred ensuring parquets exist in cache for:\n"
                             f"config={self.config}\n"
                             f"input_data_files={self.load_input_data_files}\n")
            raise
        self.log.info(f"Done caching as parquets at {self.parquet_dir}")

        kwargs = dict(self.load_dataset_kwargs)

        kwargs.pop("name", None)
        kwargs.pop("data_dir", None)

        if self.streaming:
            return datasets.load_dataset(**kwargs, streaming=True)
        return datasets.load_dataset(**kwargs, streaming=False)

    def load_secondary_data(self) -> None:
        ...

    def _ensure_parquets(self) -> None:
        os.makedirs(self.parquet_dir, exist_ok=True)

        data_files = self.load_input_data_files
        if not isinstance(data_files, list) or not isinstance(data_files, Sequence):
            data_files = [data_files]

        kw = dict(
            cache_dir=self.cache_dir,
            keep_in_memory=self.keep_in_memory,
            num_proc=self.num_proc
        )
        kw.update(dict(self.config.download_extract_transform_kwargs))

        with open(data_files[0], newline="", encoding="utf-8") as f:
            header = next(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        self.log.info(f"data_file={data_files[0]} header={header}")
        features = Features({name: Value("string") for name in header})

        ds = datasets.load_dataset(
            path=self.config.download_extract_file_type,
            data_files={"train": data_files},
            split="train",
            features=features,
            **kw,
        )

        split_datasets = {}
        if self.config.split_column:
            for split in ds.unique(self.config.split_column):
                split_datasets[split] = ds.filter(
                    partial(self._matches_split, split),
                    batched=True,
                    input_columns=self.config.split_column
                )
        else:
            split_datasets["train"] = ds

        for split, split_ds in split_datasets.items():
            if self.config.split_column_mappings is not None and split in self.config.split_column_mappings:
                split = self.config.split_column_mappings[split]

            os.makedirs(self.parquet_dir / split, exist_ok=True)

            output_parquet_file = self.parquet_dir / split / f"{split}.parquets.complete"
            if os.path.exists(output_parquet_file):
                self.log.info(f"parquets already exist: output_parquet_file={output_parquet_file}")
                continue

            for file in self.data_files[split]:
                Path(file).unlink()

            if self.config.select_columns_from_extracted is not None:
                split_ds = split_ds.select_columns(self.config.select_columns_from_extracted)

            split_ds = split_ds.map(
                self.transform_batch,
                batched=True,
                batch_size=self.config.batch_size,
                num_proc=self.config.num_proc,
            )
            split_ds = self.cast_dirs(split_ds)

            split_ds = split_ds.with_format("arrow").map(embed_table_storage, batched=True).with_format(None)

            save_parquet_shards(
                split_ds,
                self.parquet_dir / split,
                self.config.cached_parquet_size_mb,
                prefix=split
            )

            leftover = [f["filename"] for f in split_ds.cache_files]
            split_ds.cleanup_cache_files()
            for f in leftover:
                Path(f).unlink(missing_ok=True)

    @staticmethod
    def _matches_split(split: str, batch: list[str]) -> list[bool]:
        return [x == split for x in batch]

    @abstractmethod
    def cast_dirs(self, dataset: Dataset) -> Dataset:
        ...

    @abstractmethod
    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...


class MediaCachedDatasetProvider[C=MediaCachedDatasetConfig](CachedDatasetProvider[C], ABC):
    @property
    def media_dir_path(self) -> Path:
        return (self.dir / self.config.media_dir).expanduser().resolve()

    def cast_dirs(self, dataset: Dataset) -> Dataset:
        return dataset.cast_column(self.config.media_column, self.config.media_type)

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch.update({
            self.config.media_column: [str(self.media_dir_path / p) for p in batch[self.config.media_path_column]]
        })
        return batch
