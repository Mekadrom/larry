import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, DatasetDict

from larry.common.config.dataset_config import DatasetConfig
from larry.common.config.pipeline_config import PipelineConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing.preprocessing_registries import dataset_provider_config_classes_by_modality, \
    preprocessor_config_classes, default_preprocessor_config_classes_by_mapper_class, preprocessor_classes, \
    dataset_provider_classes_by_name
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.parquet_utils import save_parquet_shards
from larry.utils.types import Modality


class Pipeline:
    config: PipelineConfig
    output_dir: Path
    dataset_provider: DatasetProvider
    pipeline: list[Preprocessor]

    def __init__(self, output_dir: Path, command: Modality, pipeline_file: Path, clean: bool = False) -> None:
        self.log = logging.getLogger(__name__)

        """Instantiates a ``Runner`` and the configs specified for this preprocessing pipeline run."""
        self.log.info(f"Loading pipeline file from yaml pipeline_file={pipeline_file}")
        definition = self._load_definition_from_yaml(pipeline_file)
        self.log.info("Pipeline file loaded.")

        pipeline_config_dict = definition.get("preprocessor")
        self.config = PipelineConfig.from_dict(pipeline_config_dict)

        dataset = definition.get("dataset")
        dataset_config_name = dataset.get("config")
        dataset_config_overrides = dataset.get("overrides", [])
        dataset_config_type = dataset_provider_config_classes_by_modality[command][dataset_config_name]

        self.log.info(f"{dataset_provider_config_classes_by_modality[command]}[{dataset_config_name}]")

        self.log.info(f"Using dataset_config_name={dataset_config_name}, retrieved type: {dataset_config_type}.")

        self.log.info(f"Applying dataset_config_overrides={dataset_config_overrides}")
        dataset_provider_config: DatasetConfig = dataset_config_type(
            **{i["name"]: i["value"] for i in dataset_config_overrides}
        )

        dataset_provider_config.cache_dir = Path(dataset_provider_config.cache_dir).expanduser().resolve()
        if clean:
            self.log.info(f"cleaning up {dataset_provider_config.cache_dir}")
            shutil.rmtree(dataset_provider_config.cache_dir, ignore_errors=True)

        provider_name = dataset_provider_config.provider_name
        self.log.info(f"Using provider_name={provider_name} from config of type {type(dataset_provider_config)}")
        self.dataset_provider = dataset_provider_classes_by_name[provider_name](dataset_provider_config)

        preprocessor_configs = definition.get("pipeline")
        self.pipeline = self._build_mapper_pipeline(preprocessor_configs)

        self.output_dir = output_dir

        self.log.info("Preprocessor constructed.")

    def _load_definition_from_yaml(self, yaml_file: Path) -> dict[str, Any]:
        with open(yaml_file) as stream:
            try:
                return yaml.safe_load(stream)
            except:
                self.log.exception(f"Error loading pipeline file {yaml_file}:")
                raise

    def _build_mapper_pipeline(self, preprocessor_configs: list[dict[str, Any]]) -> list[Preprocessor]:
        pipeline: list[Preprocessor] = []

        for preprocessor_config in preprocessor_configs:
            preprocessor_class_name = preprocessor_config.pop("name")
            preprocessor_config_name = preprocessor_config.pop("config", None)
            self.log.info(
                f"Processing preprocessor_config_name={preprocessor_config_name} "
                f"for preprocessor_class_name={preprocessor_class_name}"
            )

            config_overrides = preprocessor_config.get("inputs", [])
            config_overrides = {i["name"]: i["value"] for i in config_overrides}

            preprocessor_config_instance = preprocessor_config_classes.get(
                preprocessor_config_name,
                None
            )(**config_overrides)

            if preprocessor_config_instance is None:
                preprocessor_config_instance = default_preprocessor_config_classes_by_mapper_class[
                    preprocessor_class_name
                ]

            preprocessor_config_instance.load_overrides(config_overrides)

            self.log.info(
                f"preprocessor_class_name={preprocessor_class_name} "
                f"using preprocessor_config_instance={preprocessor_config_instance}"
            )

            pipeline.append(preprocessor_classes[preprocessor_class_name](preprocessor_config_instance))

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

        self.log.info("Formatting dataset for torch...")
        dataset = dataset.with_format("torch")
        self.log.info("Dataset torch formatted.")

        torch.set_num_threads(1)

        os.makedirs(self.output_dir, exist_ok=True)
        if isinstance(dataset, DatasetDict):
            for split, split_ds in dataset.items():
                self._preprocess_and_save_shards(split, split_ds)
        elif isinstance(dataset, Dataset):
            self._preprocess_and_save_shards("train", dataset)
        else:
            raise TypeError(f"expected some kind of Dataset, got {type(dataset).__name__}")

        self._compile_and_save_provenance()
        self.log.info("Dataset saved to disk.")

    def _preprocess_and_save_shards(self, prefix: str, dataset: Dataset) -> None:
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
        save_parquet_shards(dataset, self.output_dir, self.config.parquet_size_mb, prefix)

    def _compile_and_save_provenance(self) -> None:
        """Saves a separate json file that records the sources of this saved dataset."""
        (self.output_dir / "provenance.json").write_text(
            json.dumps({"provenanceColumns": self.config.provenance_columns})
        )
