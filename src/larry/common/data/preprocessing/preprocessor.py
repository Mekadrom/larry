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
from larry.common.config.preprocess_config import PreprocessConfig
from larry.common.data.dataset_providers import DatasetProvider
from larry.common.data.preprocessing.preprocessing_mappers import PreprocessingMapper
from larry.common.data.preprocessing.preprocessing_registries import dataset_provider_config_classes_by_modality, \
    mapper_config_classes, default_mapper_config_classes_by_mapper_class, mapper_classes, \
    dataset_provider_classes_by_name
from larry.parquet_utils import save_parquet_shards
from larry.utils.types import Modality

log = logging.getLogger(__name__)


class Preprocessor:
    """The preprocessor class, which simply pipes a given dataset through a ``BatchedDatasetMapper``."""

    config: PreprocessConfig
    output_dir: Path
    dataset_provider: DatasetProvider
    pipeline: list[PreprocessingMapper]

    def __init__(self, output_dir: Path, command: Modality, pipeline_file: Path, clean: bool = False) -> None:
        """Instantiates a ``Runner`` and the configs specified for this preprocessing pipeline run."""
        log.info(f"Loading pipeline file from yaml pipeline_file={pipeline_file}")
        definition = self._load_definition_from_yaml(pipeline_file)
        log.info("Pipeline file loaded.")

        preprocessor_config_dict = definition.get("preprocessor")
        self.config = PreprocessConfig.from_dict(preprocessor_config_dict)

        dataset = definition.get("dataset")
        dataset_config_name = dataset.get("config")
        dataset_config_overrides = dataset.get("overrides", [])
        dataset_config_type = dataset_provider_config_classes_by_modality[command][dataset_config_name]

        log.info(f"{dataset_provider_config_classes_by_modality[command]}[{dataset_config_name}]")

        log.info(f"Using dataset_config_name={dataset_config_name}, retrieved type: {dataset_config_type}.")

        log.info(f"Applying dataset_config_overrides={dataset_config_overrides}")
        dataset_provider_config: DatasetConfig = dataset_config_type(
            **{i["name"]: i["value"] for i in dataset_config_overrides}
        )

        dataset_provider_config.cache_dir = Path(dataset_provider_config.cache_dir).expanduser().resolve()
        if clean:
            log.info(f"cleaning up {dataset_provider_config.cache_dir}")
            shutil.rmtree(dataset_provider_config.cache_dir, ignore_errors=True)

        provider_name = dataset_provider_config.provider_name
        log.info(f"Using provider_name={provider_name} from config of type {type(dataset_provider_config)}")
        try:
            self.dataset_provider = dataset_provider_classes_by_name[provider_name](
                dataset_provider_config
            )
        except KeyError:
            log.critical(f"key {provider_name} not found in {dataset_provider_classes_by_name.keys()}")
            raise

        pipeline_config = definition.get("pipeline")
        self.pipeline = self._build_mapper_pipeline(pipeline_config)

        self.output_dir = output_dir

        log.info("Preprocessor constructed.")

    @staticmethod
    def _load_definition_from_yaml(yaml_file: Path) -> dict[str, Any]:
        with open(yaml_file) as stream:
            try:
                return yaml.safe_load(stream)
            except:
                log.exception(f"Error loading pipeline file {yaml_file}:")
                raise

    @staticmethod
    def _build_mapper_pipeline(config: dict[str, Any]) -> list[PreprocessingMapper]:
        pipeline: list[PreprocessingMapper] = []

        for mapper_section in config.get("mappers"):
            mapper_class_name = mapper_section.pop("name")
            log.info(f"Processing mapper_config for mapper_class_name={mapper_class_name}")

            mapper_config_name = mapper_section.pop("config", None)

            config_overrides = mapper_section.get("inputs", [])
            config_overrides = {i["name"]: i["value"] for i in config_overrides}

            preprocess_config = mapper_config_classes.get(
                mapper_config_name,
                None
            )(**config_overrides)

            if preprocess_config is None:
                preprocess_config = default_mapper_config_classes_by_mapper_class[mapper_class_name]

            preprocess_config.load_overrides(config_overrides)

            log.info(f"mapper_class_name={mapper_class_name} using preprocess_config={preprocess_config}")

            pipeline.append(mapper_classes[mapper_class_name](preprocess_config))

        return pipeline

    def run(self) -> None:
        """Runs preprocessing."""
        log.info("Beginning preprocessing...")
        try:
            self.preprocess_all()
            log.info("Preprocessing finished.")
        except:
            log.exception("Error during preprocessing:")
            sys.exit(-2)

    def preprocess_all(self) -> None:
        """Super method for the primary entrypoint for preprocessors. Does all the work using ``self.config``."""
        log.info("Initializing dataset...")
        dataset = self.dataset_provider.load_dataset()
        log.info("Dataset initialized. Preprocessing...")

        log.info("Formatting dataset for torch...")
        dataset = dataset.with_format("torch")
        log.info("Dataset torch formatted.")

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
        log.info("Dataset saved to disk.")

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
