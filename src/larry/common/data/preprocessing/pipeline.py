import glob
import json
import logging
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from larry.common.config.config import RunRecord
from larry.common.config.data.preprocessing import pipeline_config
from larry.common.data.utils import save_parquet_shards


class Pipeline:
    """Management class for preprocessing pipelines; handles configuration loading/parsing and final output saving."""

    output_dir: Path
    state: pipeline_config.PipelineState
    provenance_only: bool

    def __init__(
            self,
            definition: pipeline_config.PipelineDefinition,
            output_dir: Path,
            clean: bool = False,
            provenance_only: bool = False,
    ) -> None:
        """Sets up the pipeline run."""
        self.log = logging.getLogger(type(self).__name__)

        self.record = RunRecord.of(definition)
        self.state = pipeline_config.PipelineState(definition)

        self.output_dir = output_dir

        if clean:
            self._clean(self.state.dataset_provider.config.cache_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        self.provenance_only = provenance_only

    def _clean(self, dataset_provider_cache_dir: str | Path | Literal[""] | None) -> None:
        """Cleans up parquet dirs, *.complete marker files, provenance.json, and cache dirs."""
        if self.output_dir.exists():
            self.log.info(f"Cleaning up {self.output_dir}")

            # removes split dirs containing parquets
            subdirs = [x for x in self.output_dir.iterdir() if x.is_dir()]
            for subdir in subdirs:
                shutil.rmtree(subdir, ignore_errors=True)

            # to be re-generated
            (self.output_dir / "provenance.json").unlink(missing_ok=True)

            # remove markers
            for completed_marker in glob.glob(str(self.output_dir / "*.complete")):
                Path(completed_marker).resolve().unlink()

        if dataset_provider_cache_dir:
            self.log.info(f"Cleaning up {dataset_provider_cache_dir}")
            shutil.rmtree(dataset_provider_cache_dir, ignore_errors=True)

    def run(self) -> None:
        """Runs the preprocessing pipeline."""
        self.log.info("Beginning preprocessing...")
        try:
            self.preprocess_all()
            self.log.info("Preprocessing finished.")
        except Exception:
            self.log.exception("Error during preprocessing:")
            sys.exit(-2)

    def preprocess_all(self) -> None:
        """The primary entrypoint for preprocessors."""
        if self.provenance_only:
            self.log.warning("--provenance_only passed; will skip preprocessing.")
            self._compile_and_save_provenance()
            return

        self.log.info("Initializing dataset...")
        dataset = self.state.dataset_provider.load_dataset()
        self.log.info("Dataset initialized. Preprocessing...")

        torch.set_num_threads(1)

        written: set[str] = set()
        if isinstance(dataset, DatasetDict):
            for split_name, split_ds in dataset.items():
                self._preprocess_and_save_shards(split_ds, str(split_name), written)
        elif isinstance(dataset, Dataset):
            self._preprocess_and_save_shards(dataset, "train", written)
        else:
            raise TypeError(f"expected some kind of Dataset, got {type(dataset).__name__}")

        # save after preprocessing is finished because pipeline steps may mutate provenance information
        self._compile_and_save_provenance()

        self.log.info("Dataset saved to disk.")

    def _preprocess_and_save_shards(self, dataset: Dataset | DatasetDict, input_split: str, written: set[str]) -> None:
        dataset = self._preprocess(dataset, input_split, written)
        self._save_shards(dataset)

    def _preprocess(self, dataset: Dataset | DatasetDict, input_split: str, written: set[str]) -> DatasetDict:
        for mapper in self.state.pipeline:
            try:
                mapper.log.info(f"columns in: {dataset.column_names}")

                if isinstance(dataset, Dataset):
                    before = len(dataset)
                elif isinstance(dataset, DatasetDict):
                    before = {k: len(d) for k, d in dataset.items()}

                dataset = mapper.preprocess_dataset(dataset)

                if isinstance(dataset, Dataset):
                    mapper.log.info(f"num_seen={before}, num_out={len(dataset)}")
                elif isinstance(dataset, DatasetDict):
                    log_dataset_dict = {k: len(d) for k, d in dataset.items()}
                    mapper.log.info(f"num_seen={before}, num_out={log_dataset_dict}")

                mapper.log.info(f"columns out: {dataset.column_names}")
            except:
                mapper.log.critical(f"Error during _preprocess_and_save_shards in mapper={mapper}")
                raise

        if isinstance(dataset, Dataset):
            if self.state.fractions is not None:
                dataset = self._split_by_fractions(dataset, self.state.fractions)
            else:
                dataset = DatasetDict({input_split: dataset})

        names = [str(name) for name in dataset]
        collisions = sorted(written.intersection(names))
        if collisions:
            raise ValueError(
                f"input split {input_split!r} would write output splits {collisions} that an earlier input split "
                f"already wrote; configure the pipeline so every output split name is produced once"
            )
        written.update(names)
        return dataset

    @staticmethod
    def _split_by_fractions(dataset: Dataset, fractions: Mapping[str, float]) -> DatasetDict:
        """Splits a given dataset into different splits depending on fraction definitions.

        Takes fractions like {"train": 0.98, "val": 0.01, "test": 0.01}, produces an array of [0.0, 0.98, 0.99, 1.00]
        and then multiplies to the length of the dataset to get something like [0, 9800, 9900, 10000] (for a dataset
        of length 10,000) representing each bound of a ranged indices for each split requested (has to cast to int for
        rounding). Returns a ``DatasetDict``.
        """
        n = len(dataset)

        indices = np.cumsum([0.0, *fractions.values()]) * n
        indices = np.round(indices).astype(int)
        indices[-1] = n
        return DatasetDict({
            name: dataset.select(range(int(a), int(b)))
            for name, a, b in zip(fractions, indices[:-1], indices[1:], strict=True)
        })

    def _save_shards(self, dataset: DatasetDict) -> None:
        """Saves each split to separate parquet dirs."""
        for split_name, split_dataset in dataset.items():
            save_parquet_shards(
                split_dataset,
                self.output_dir,
                self.state.config.parquet_size_mb,
                prefix=str(split_name),
                writer_batch_size=self.state.config.default_batch_size,
            )

    def _compile_and_save_provenance(self) -> None:
        """Saves a separate json file that records the sources of this saved dataset."""
        provenance = self.state.provenance
        provenance.populate_mutables()
        provenance.config = self.record.config
        provenance.cli = self.record.cli
        (self.output_dir / "provenance.json").write_text(json.dumps(provenance.to_dict(), indent="\t"))
        self.record.write(self.output_dir)
