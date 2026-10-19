from abc import ABC, abstractmethod
from typing import Any

import torch.cuda
from datasets import Dataset, DatasetDict, Features, concatenate_datasets
from datasets.features.features import FeatureType

from larry.common.config.data.preprocessing.mapper_configs import (
    CompositeIDMapperConfig,
    MapperConfig,
    SingleColumnMapperConfig,
    ValueOverrideMapperConfig,
)
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Mapper[C: MapperConfig = MapperConfig](Preprocessor[C], ABC):
    """Superclass for all ``Preprocessor`` instances that map from N input columns to M output columns."""

    def __init__(self, provenance: Provenance, config: C) -> None:
        """Sets up all ``Mapper`` instances.

        Sets up with a ``self.remove_columns`` for columns to remove columns after this step runs,
        ``self.input_columns``, and ``self.output_columns`` filtered for None values.
        """
        super().__init__(provenance, config)
        if self.config.remove_columns:
            self.remove_columns = [c for c in self.config.remove_columns if c is not None]
        else:
            self.remove_columns = []
        self.input_columns = [c for c in self.config.input_columns if c is not None]
        self.output_columns = [c for c in self.config.output_columns if c is not None]
        self.device_index = 0

    def validate(self) -> None:
        """Simple override validating the input config; called by the super class, ``Preprocessor`` in init."""
        if not self.config.input_columns:
            raise ValueError(f"input_columns must be specified for {self.__class__.__name__}")
        if not self.config.output_columns:
            raise ValueError(f"output_columns must be specified for {self.__class__.__name__}")

    def map_kwargs(self) -> dict[str, Any]:
        """Overrideable function for returning kwargs passed to the default ``preprocess_dataset`` function.

        Some amount of sanitization should be done on this prior to allowing it to be passed directly to
        ``dataset.map``.
        """
        return dict(
            batched=True,
            batch_size=self.config.batch_size,
            writer_batch_size=self.config.writer_batch_size,
            num_proc=self.config.num_proc,
            remove_columns=self.remove_columns,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
        )

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """For a ``Dataset`` or ``DatasetDict``, calls ``_preprocess_dataset`` for each underlying ``Dataset``."""
        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict({
                split: self._preprocess_dataset(split_ds)
                for split, split_ds in dataset.items()
            })
        else:
            dataset = self._preprocess_dataset(dataset)
        return self._filter_nulls(dataset)

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Real generic entrypoint for mapping the input dataset.

        Provides all configured input columns and output features from the overridden ``output_features`` method, and
        uses the return value of ``map_kwargs`` in conjunction with an overridden remove_columns value. Passes device
        rank as well for multi-gpu processing of gpu-parallelizable mapper steps. To avoid passing all columns of a
        dataset as materialized data, the actual map call expects the implementation of mapper to only return the
        columns to be added, with no rows removed. It then concatenates the new columns onto the existing dataset,
        only requiring that the ``self.input_columns`` be loaded into memory for this step's processing. Additionally
        flattens indices if required prior to concatenation.
        """
        out = dataset.map(
            self._preprocess_columns,
            input_columns=self.input_columns,
            features=self.output_features(dataset),
            with_rank=True,
            **{**self.map_kwargs(), "remove_columns": dataset.column_names},
        )

        # prior to concatenating the output columns, make sure they don't already exist in the dataset
        kept = dataset.remove_columns([
            c
            for c in dataset.column_names
            if c in out.column_names or c in self.remove_columns
        ])

        # noinspection PyProtectedMember
        if kept._indices is not None:
            kept = kept.flatten_indices(num_proc=self.config.num_proc, writer_batch_size=self.config.writer_batch_size)

        return concatenate_datasets([kept, out], axis=1)

    def _preprocess_columns(self, *args: Any) -> dict[str, list[Any]]:
        """Helper/wrapper for preprocessing a batch.

        Since specifying multiple input columns provides them to the batch function as ordered args, this takes them and
        pairs them with the column_name to create a dict batch to provide to the _preprocess_batch function instead.
        """
        *columns, rank = args
        return self._preprocess_batch(dict(zip(self.input_columns, columns, strict=True)), rank)

    def _preprocess_batch(self, batch: dict[str, list[Any]], rank: int | None) -> dict[str, list[Any]]:
        """Wrapper/helper that makes sure this mapper instance uses the correct device according to the device index.

        Avoids requiring all subclasses to care about such things.
        """
        if rank is not None:
            n_devices = torch.cuda.device_count()
            if n_devices > 0:
                self.device_index = rank % n_devices
        return self.preprocess_batch(batch)

    @abstractmethod
    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Abstract entrypoint for batch-based preprocessing.

        The input ``batch`` is a ``dict[str, list[Any]]``, where the string keys are input column names and the list
        values are the values per-row in order for each example in each column. Can be iterated safely in order over
        using ``zip``.
        """
        ...

    def _filter_nulls(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """After preprocessing the entire dataset by batches, filter for nulls.

        Filters rows that have null values in any of the output columns.
        """
        if isinstance(dataset, DatasetDict):
            for split, split_ds in dataset.items():
                dataset[split] = self.filter_nulls(split_ds)
        elif isinstance(dataset, Dataset):
            dataset = self.filter_nulls(dataset)
        return dataset

    def filter_nulls(self, dataset: Dataset) -> Dataset:
        """Checks to see if any nulls are present in the output columns after this mapper preprocessed.

        Then, does a specialized prune step that removes any rows that have any nulls in the output columns.
        """
        n = sum(dataset.data.column(c).null_count for c in self.output_columns)
        if self.config.filter_null_outputs and n > 0:
            return dataset.filter(
                lambda *cols: [all(v is not None for v in row) for row in zip(*cols, strict=True)],
                input_columns=self.output_columns,
                batched=True,
                batch_size=self.config.batch_size,
                writer_batch_size=self.config.writer_batch_size,
                keep_in_memory=not self.config.cache_results,
                load_from_cache_file=self.config.cache_results,
            )
        return dataset

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        """Overrideable method to return a ``Features`` object (dict) mapping output column names to ``FeatureType``.

        Returning ``None`` lets datasets guess what the type is, which is relatively unsafe and not recommended.
        """
        return None


class SingleColumnMapper[I, O, C: SingleColumnMapperConfig = SingleColumnMapperConfig](Mapper[C], ABC):
    """Subclass of ``Mapper`` for mappers that operate on a single input column and produce a single output column."""

    def validate(self) -> None:
        """Validates the singular input and output column names are configured."""
        super().validate()
        if not self.config.input_column:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if not self.config.output_column:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Helper/wrapper for preprocess_batch.

        Takes the required single input column values and passes it example-wise to a single-item preprocessing
        function. Produces the single output column as an ordered list of corresponding values.
        """
        column_in = batch[self.config.input_column]
        column_out = []
        for example in column_in:
            column_out.append(self.preprocess_example(example))
        return {self.config.output_column: column_out}

    def preprocess_example(self, example: I) -> O:
        """Takes a single input value as a value from the input column and produces a corresponding output."""
        ...

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        """Helper/wrapper for a single output feature. Only the one output feature needs to be specified."""
        output_feature = self.output_feature()
        if output_feature is None:
            return None
        return Features({self.config.output_column: output_feature})

    @abstractmethod
    def output_feature(self) -> FeatureType | None:
        """Simplified abstract method for specifying a single output ``FeatureType``."""
        ...


class ValueOverrideMapper(Mapper[ValueOverrideMapperConfig]):
    """Mapper implementation overrides specific values."""

    def validate(self) -> None:
        """Validates the input and output columns in addition to configured value mappings."""
        super().validate()
        if not self.config.mappings:
            raise ValueError(f"mappings_config must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Batch preprocessor for overriding specific row values.

        Matches on a single input column's given value, sets the value of the output column for that row to the given
        override value. Does this for any number of configured mappings.
        """
        output_batch = {}

        def current(column: str) -> list[Any]:
            """Simple helper function for getting the current value of a column.

            Retrieves the value from either the input batch or the WIP output batch.
            """
            return output_batch.get(column, batch[column])

        for rule in self.config.mappings:
            source = current(rule.input_column)
            target = output_batch.setdefault(rule.output_column, list(batch[rule.output_column]))
            for i, value in enumerate(source):
                if value == rule.input_column_value:
                    target[i] = rule.output_column_value
        return output_batch


class CompositeIDMapper(Mapper[CompositeIDMapperConfig]):
    """Mapper that takes one or more input columns and produces a composite ID column."""

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Batch preprocessor that generates a composite ID for each example.

        Takes one or more input columns, joins them with a configured delimiter, and produces a new column with the
        joined value.
        """
        input_columns = [batch[c] for c in self.config.input_columns]
        return {
            self.config.id_output_column: [
                self.config.delim.join(str(i) for i in example_inputs)
                for example_inputs in zip(*input_columns, strict=True)
            ]
        }
