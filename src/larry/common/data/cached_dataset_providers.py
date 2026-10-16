import glob
import logging
import os.path
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import datasets
from datasets import Dataset, DatasetDict, Features, IterableDataset, IterableDatasetDict
from datasets.table import embed_table_storage

from larry.common.config.data.preprocessing.dataset_configs import CachedDatasetConfig, MediaCachedDatasetConfig
from larry.common.data.dataset_providers import HuggingFaceDatasetProvider
from larry.common.data.downloaders import Downloader
from larry.common.data.extractors import Extractor
from larry.common.data.utils import save_parquet_shards


class CachedDatasetProvider[C: CachedDatasetConfig = CachedDatasetConfig](HuggingFaceDatasetProvider[C], ABC):
    """A ``DatasetProvider`` for datasets that need to be downloaded and extracted from various formats manually.

    Essentially replaces arbitrary python code execution scripts in datasets that still require them.
    """

    def __init__(self, config: C) -> None:
        """Sets up cache and save dirs."""
        super().__init__(config)

        self.download_extract_cache_dir = Path(self.config.download_extract_cache_dir).expanduser().resolve()
        self.parquet_dir = self.download_extract_cache_dir / "parquets"

    @property
    def path(self) -> str:
        """Override of the dataset path for the output dataset to "parquet", which is a special term in datasets."""
        return "parquet"

    @property
    def data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]]:
        """Builds a mapping of {split: [parquet_globs...]} for the output of this dataset provider."""
        return {
            split: glob.glob(str(self.parquet_dir / f"{split}/{split}-*.parquet"))
            for split in os.listdir(self.parquet_dir)
            if self._is_complete(split)
        }

    @property
    @abstractmethod
    def load_input_data_files(self) -> Mapping[str, str]:
        """Returns a mapping of {split: [input_files_for_processing...]} as input to this dataset provider."""
        ...

    def load_dataset(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Primary entrypoint for ``CachedDatasetProviders``.

        Does all downloads, followed by all extractions, followed by loading "secondary data", then finally saves the
        dataset as parquets with proper column and types. Each step along the way should support caching and resuming in
        some manner.
        """
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
        """Performs all downloads in order. Downloaders are instantiated and ran sequentially."""
        for impl, config in self.config.downloader_configs():
            impl_type = Downloader.REGISTRY.resolve(impl)
            impl_type(config, self.download_extract_cache_dir).ensure_downloaded()

    def extract(self) -> None:
        """Performs all extractions in order. Extractors are instantiated and ran sequentially."""
        for impl, config in self.config.extractor_configs():
            impl_type = Extractor.REGISTRY.resolve(impl)
            impl_type(config, self.download_extract_cache_dir).ensure_extracted()

    def load_secondary_data(self) -> None:
        """Optional secondary data.

        Initial preprocessing, additional downloads post-extraction, or any other data manipulation that otherwise
        couldn't occur in the parquet saving step.
        """
        ...

    def _ensure_parquets(self) -> None:
        """Loads the dataset input files from one of the supported non-parquet dataset file types

        Supports input files in the formats CSV, JSONL (maybe more) and transforms the dataset using overridden
        functionality to contain the materialized data for the provider to save as parquets for caching and to also
        provide as the dataset for either preprocessing or training.
        """
        os.makedirs(self.parquet_dir, exist_ok=True)

        data_files = self.load_input_data_files
        if self.config.split_column and len(data_files) != 1:
            raise ValueError(
                f"split_column={self.config.split_column!r} splits a single input file, "
                f"but load_input_data_files names {sorted(data_files)}"
            )

        for input_split, file in data_files.items():
            if not self.config.split_column and self._is_complete(input_split):
                self.log.info(f"{input_split} parquets already exist in {self.parquet_dir}")
                continue
            if not Path(file).exists():
                raise FileNotFoundError(f"input data file for split {input_split!r} does not exist: {file}")

            try:
                ds = datasets.load_dataset(
                    path=self.config.download_extract_file_type,
                    data_files={input_split: file},
                    split=input_split,
                    cache_dir=str(self.cache_dir),
                    features=self.make_features(data_files),
                    keep_in_memory=self.keep_in_memory,
                    streaming=False,
                    num_proc=self.num_proc,
                    **self.config.download_extract_transform_kwargs,
                )
            except Exception:
                self.log.error(
                    f"error loading cached dataset from file: path={self.config.download_extract_file_type} "
                    f"data_files={{{input_split}: {file}}} split={input_split} cache_dir={str(self.cache_dir)}"
                )
                raise

            split_datasets = {}
            if self.config.split_column:
                for split in ds.unique(self.config.split_column):
                    split_datasets[split] = ds.filter(
                        partial(self._matches_split, split),
                        batched=True,
                        input_columns=self.config.split_column
                    )
            else:
                split_datasets[input_split] = ds

            # here is where transforming and saving is delegated out from here
            for split, split_ds in split_datasets.items():
                if self.config.split_column_mappings is not None and split in self.config.split_column_mappings:
                    split = self.config.split_column_mappings[split]
                self._write_split(split, split_ds)

    def _is_complete(self, split: str) -> bool:
        """Simple helper for determining if a split's complete marker file is present."""
        return (self.parquet_dir / f"{split}.parquets.complete").exists()

    def _write_split(self, split: str, split_ds: Dataset) -> None:
        """Does the mapping of a split using ``transform_batch`` and saves to disk."""
        if self._is_complete(split):
            self.log.info(f"{split} parquets already exist in {self.parquet_dir}")
            return

        split_dir = self.parquet_dir / split
        os.makedirs(split_dir, exist_ok=True)

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
            prefix=split,
            writer_batch_size=self.config.batch_size
        )

        leftover = [f["filename"] for f in split_ds.cache_files]
        split_ds.cleanup_cache_files()
        for f in leftover:
            Path(f).unlink(missing_ok=True)

    def make_features(self, data_files: Mapping[str, str]) -> Features | None:
        """None will cause datasets to intuit the output features, it is recommended to override to clarify."""
        return None

    @staticmethod
    def _matches_split(split: str, batch: list[str]) -> list[bool]:
        """Fingerprintable non-lambda for filtering a dataset for matching a given split name in a specific column."""
        return [x == split for x in batch]

    @abstractmethod
    def cast_cols(self, dataset: Dataset) -> Dataset:
        """Convenience method for casting columns separately, even after make_features. TODO: remove this"""
        ...

    @abstractmethod
    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """For transforming the raw TSV/CSV/whatever dataset into the materialized form."""
        ...


class MediaCachedDatasetProvider[C: MediaCachedDatasetConfig = MediaCachedDatasetConfig](CachedDatasetProvider[C], ABC):
    """Specialization of ``CachedDatasetProvider`` for datasets that have specific media columns like audio or image."""

    @property
    def media_dir_path(self) -> Path:
        """Default implementation for saving media files to cache prior to saving into parquets as bytes."""
        return (self.download_extract_cache_dir / self.config.media_dir).expanduser().resolve()

    def cast_cols(self, dataset: Dataset) -> Dataset:
        """Overridden to cast the output media column to the configured FeatureType."""
        return dataset.cast_column(self.config.media_column, self.config.media_type)

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Given the paths in media_path_column are relative to the media_dir_path configured, resolves those paths."""
        batch.update({
            self.config.media_column: [str(self.media_dir_path / p) for p in batch[self.config.media_path_column]]
        })
        return batch
