import glob
import logging
import os.path
from abc import abstractmethod, ABC
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import datasets
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset, Features
from datasets.table import embed_table_storage

from larry.common.config.data.preprocessing.dataset_configs import CachedDatasetConfig, MediaCachedDatasetConfig
from larry.common.data.dataset_providers import HuggingFaceDatasetProvider
from larry.common.data.downloaders import Downloader
from larry.common.data.extractors import Extractor
from larry.common.data.utils import save_parquet_shards


class CachedDatasetProvider[C: CachedDatasetConfig = CachedDatasetConfig](HuggingFaceDatasetProvider[C], ABC):
    def __init__(self, config: C) -> None:
        super().__init__(config)

        self.download_extract_cache_dir = Path(self.config.download_extract_cache_dir).expanduser().resolve()
        self.parquet_dir = self.download_extract_cache_dir / "parquets"

    @property
    def path(self) -> str:
        return "parquet"

    @property
    def data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]]:
        return {
            split: glob.glob(str(self.parquet_dir / f"{split}/{split}-*.parquet"))
            for split in os.listdir(self.parquet_dir)
        }

    @property
    @abstractmethod
    def load_input_data_files(self) -> Mapping[str, str]:
        ...

    def load_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        os.makedirs(self.download_extract_cache_dir, exist_ok=True)

        self.log.info("Downloading...")
        self.download()
        self.log.info("Done downloading.")

        self.log.info("Extracting...")
        self.extract()
        self.log.info("Done extracting.")

        self.load_secondary_data()

        self.log.info("Caching as parquets...")
        try:
            self._ensure_parquets()
        except Exception:
            logging.critical(
                f"Exception occurred ensuring parquets exist in cache for:\n"
                f"config={self.config}\n"
                f"input_data_files={self.load_input_data_files}\n"
            )
            raise
        self.log.info(f"Done caching as parquets at {self.parquet_dir}")

        kwargs = dict(self.load_dataset_kwargs)

        kwargs.pop("name", None)
        kwargs.pop("data_dir", None)

        if self.streaming:
            return datasets.load_dataset(**kwargs, streaming=True)
        return datasets.load_dataset(**kwargs, streaming=False)

    def download(self) -> None:
        for impl, config in self.config.downloader_configs():
            impl_type = Downloader.REGISTRY.resolve(impl)
            impl_type(config, self.download_extract_cache_dir).ensure_downloaded()

    def extract(self) -> None:
        for impl, config in self.config.extractor_configs():
            impl_type = Extractor.REGISTRY.resolve(impl)
            impl_type(config, self.download_extract_cache_dir).ensure_extracted()

    def load_secondary_data(self) -> None:
        ...

    def _ensure_parquets(self) -> None:
        os.makedirs(self.parquet_dir, exist_ok=True)

        data_files = self.load_input_data_files
        ds = datasets.load_dataset(
            path=self.config.download_extract_file_type,
            data_files={"train": data_files["train"]},
            split="train",
            cache_dir=str(self.cache_dir),
            features=self.make_features(data_files),
            keep_in_memory=self.keep_in_memory,
            streaming=False,
            num_proc=self.num_proc,
            **self.config.download_extract_transform_kwargs,
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

            split_dir = self.parquet_dir / split
            os.makedirs(split_dir, exist_ok=True)

            output_parquet_file = self.parquet_dir / f"{split}.parquets.complete"
            if os.path.exists(output_parquet_file):
                self.log.info(f"{split} parquets already exist: output_parquet_file={output_parquet_file}")
                continue

            # parquest.complete file was not created, so this is potentially a partial resume. delete existing parquets
            # in the target dir
            for f in split_dir.glob("*.parquet"):
                f.unlink()

            if self.config.select_columns_from_extracted is not None:
                split_ds = split_ds.select_columns(self.config.select_columns_from_extracted)

            split_ds = split_ds.map(
                self.transform_batch,
                batched=True,
                batch_size=self.config.batch_size,
                num_proc=self.config.num_proc,
            )
            split_ds = self.cast_cols(split_ds)
            split_ds = split_ds.with_format("arrow").map(embed_table_storage, batched=True).with_format(None)

            save_parquet_shards(
                split_ds,
                self.parquet_dir,
                self.config.cached_parquet_size_mb,
                prefix=split
            )

            leftover = [f["filename"] for f in split_ds.cache_files]
            split_ds.cleanup_cache_files()
            for f in leftover:
                Path(f).unlink(missing_ok=True)

    def make_features(self, data_files: Mapping[str, str]) -> Features | None:
        return None

    @staticmethod
    def _matches_split(split: str, batch: list[str]) -> list[bool]:
        return [x == split for x in batch]

    @abstractmethod
    def cast_cols(self, dataset: Dataset) -> Dataset:
        ...

    @abstractmethod
    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...


class MediaCachedDatasetProvider[C: MediaCachedDatasetConfig = MediaCachedDatasetConfig](CachedDatasetProvider[C], ABC):
    @property
    def media_dir_path(self) -> Path:
        return (self.download_extract_cache_dir / self.config.media_dir).expanduser().resolve()

    def cast_cols(self, dataset: Dataset) -> Dataset:
        return dataset.cast_column(self.config.media_column, self.config.media_type)

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch.update({
            self.config.media_column: [str(self.media_dir_path / p) for p in batch[self.config.media_path_column]]
        })
        return batch
