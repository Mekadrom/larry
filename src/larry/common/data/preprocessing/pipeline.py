import json
import logging
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.dataset_configs import DatasetConfig
from larry.common.config.data.preprocessing.pipeline_config import PipelineConfig
from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.common.data.utils import save_parquet_shards, split_by_fractions
from larry.common.utils import load_yaml_config, git_utils


class Pipeline:
    config: PipelineConfig
    output_dir: Path
    dataset_provider: DatasetProvider
    pipeline: list[Preprocessor]

    def __init__(self, config_file: Path, output_dir: Path, clean: bool = False) -> None:
        """Instantiates a ``Runner`` and the configs specified for this preprocessing pipeline run."""
        self.log = logging.getLogger(__name__)
        self.output_dir = output_dir

        self.log.info(f"Loading pipeline config from yaml config_file={config_file}")
        definition = load_yaml_config(config_file)
        self.log.info("Pipeline config loaded.")

        preprocessor_config_dict: dict[str, Any] = definition.get("preprocessor", {})
        preprocessor_inputs: list[dict[str, Any]] = preprocessor_config_dict.get("inputs", [])
        self.log.info(f"using preprocessor_inputs={preprocessor_inputs}")
        self.config = PipelineConfig(
            **{i["name"]: i["value"] for i in preprocessor_inputs}
        )

        dataset_config_dict: dict[str, Any] = definition.get("dataset", {})

        dataset_config_type_name = dataset_config_dict.get("config")
        dataset_config_split_fractions = dataset_config_dict.get("splits", None)
        if dataset_config_split_fractions:
            self.fractions = {i["name"]: i["part"] for i in dataset_config_split_fractions}
        else:
            self.fractions = {"train": 1.0}

        dataset_config_type = DatasetConfig.REGISTRY.resolve(dataset_config_type_name)
        self.log.info(f"Using dataset_config_type_name={dataset_config_type_name}")

        dataset_config_overrides = dataset_config_dict.get("overrides", [])
        self.log.info(f"Applying dataset_config_overrides={dataset_config_overrides}")

        dataset_provider_config: DatasetConfig = dataset_config_type(
            **{i["name"]: i["value"] for i in dataset_config_overrides}
        )

        if dataset_provider_config.cache_dir:
            dataset_provider_config.cache_dir = Path(dataset_provider_config.cache_dir).expanduser().resolve()
            if clean:
                self.log.info(f"Cleaning up {dataset_provider_config.cache_dir}")
                shutil.rmtree(dataset_provider_config.cache_dir, ignore_errors=True)

        dataset_provider_type_name = dataset_provider_config.provider_name
        self.log.info(
            f"Using dataset_provider_type_name={dataset_provider_type_name} "
            f"for dataset_provider_config={dataset_provider_config}"
        )
        dataset_provider_type = DatasetProvider.REGISTRY.resolve(dataset_provider_type_name)
        self.dataset_provider = dataset_provider_type(dataset_provider_config)

        pipeline_configs = definition.get("pipeline", None)
        if pipeline_configs:
            self.pipeline = self._build_mapper_pipeline(pipeline_configs)
        else:
            self.log.critical(
                "No pipeline configured; parquets will be downloaded and cached locally according to the configuration."
            )
            self.pipeline = []
        self.log.info("Preprocessor constructed")

    def _build_mapper_pipeline(self, pipeline_configs: list[dict[str, Any]]) -> list[Preprocessor]:
        pipeline: list[Preprocessor] = []

        for preprocessor_config in pipeline_configs:
            preprocessor_type_name = preprocessor_config.get("name")
            preprocessor_type = Preprocessor.REGISTRY.resolve(preprocessor_type_name)

            preprocessor_config_type_name = preprocessor_config.get("config")
            preprocessor_config_type = PreprocessorConfig.REGISTRY.resolve(preprocessor_config_type_name)

            self.log.info(
                f"Using preprocessor_config_type_name={preprocessor_config_type_name} "
                f"for preprocessor_type_name={preprocessor_type_name}"
            )

            preprocessor_config_overrides = preprocessor_config.get("inputs", [])
            self.log.info(f"Applying preprocessor_config_overrides={preprocessor_config_overrides}")
            preprocessor_config = preprocessor_config_type(
                **{i["name"]: i["value"] for i in preprocessor_config_overrides}
            )

            self.log.info(
                f"preprocessor_type_name={preprocessor_type_name} "
                f"using preprocessor_config={preprocessor_config}"
            )

            pipeline.append(preprocessor_type(self.config.provenance_columns, preprocessor_config))

        return pipeline

    def run(self) -> None:
        """Runs preprocessing."""
        self.log.info("Beginning preprocessing...")
        try:
            self.preprocess_all()
            self.log.info("Preprocessing finished.")
        except:
            self.log.exception("Error during preprocessing:")
            sys.exit(-2)

    def preprocess_all(self) -> None:
        """Super method for the primary entrypoint for preprocessors. Does all the work using ``self.config``."""
        self.log.info("Initializing dataset...")
        dataset = self.dataset_provider.load_dataset()
        self.log.info("Dataset initialized. Preprocessing...")

        torch.set_num_threads(1)

        os.makedirs(self.output_dir, exist_ok=True)
        if isinstance(dataset, DatasetDict):
            for _, split_ds in dataset.items():
                self._preprocess_and_save_shards(split_ds)
        elif isinstance(dataset, Dataset):
            self._preprocess_and_save_shards(dataset)
        else:
            raise TypeError(f"expected some kind of Dataset, got {type(dataset).__name__}")

        self._compile_and_save_provenance()
        self.log.info("Dataset saved to disk.")

    def _preprocess_and_save_shards(self, dataset: Dataset | DatasetDict) -> None:
        if not math.isclose(sum(self.fractions.values()), 1.0, abs_tol=1e-6):
            raise ValueError("")

        for mapper in self.pipeline:
            try:
                mapper.log.info(f"columns in: {dataset.column_names}")
                before = len(dataset)
                dataset = mapper.preprocess_dataset(dataset)
                mapper.log.info(f"num_seen={before}, num_out={len(dataset)}")
                mapper.log.info(f"columns out: {dataset.column_names}")
            except:
                mapper.log.critical(f"Error during _preprocess_and_save_shards in mapper={mapper}")
                raise

        if self.fractions and not isinstance(dataset, DatasetDict):
            dataset = split_by_fractions(dataset, self.config.default_batch_size, self.fractions, seed=self.config.seed)

        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_name = str(split_name)
                save_parquet_shards(
                    split_dataset.map(
                        batched=True,
                        batch_size=self.config.default_batch_size,
                        writer_batch_size=self.config.default_batch_size
                    ),
                    self.output_dir,
                    self.config.parquet_size_mb,
                    prefix = split_name
                )
        else:
            save_parquet_shards(dataset, self.output_dir, self.config.parquet_size_mb, prefix="train")

    def _compile_and_save_provenance(self) -> None:
        """Saves a separate json file that records the sources of this saved dataset."""
        provenance = {
            "provenanceColumns": self.config.provenance_columns,
            "produced_on": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        repo_state = git_utils.git_info()
        if repo_state is not None:
            provenance.update({
                "git_commit": repo_state.git_commit,
                "git_is_dirty": repo_state.git_dirty,
            })
        else:
            provenance.update({
                "git_commit": "not a git repo",
                "git_is_dirty": "not a git repo"
            })

        if self.dataset_provider.config.path not in ("parquet", "csv", "json", "tsv"):
            # since the dataset provider is where this dataset came from, it is the direct ancestor. ancestors of those
            # datasets will be able to crawled by a script that has yet to be made
            provenance.update({
                "ancestry": self.dataset_provider.config.path
            })

        (self.output_dir / "provenance.json").write_text(json.dumps(provenance))
