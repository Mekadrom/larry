import json
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import jsonpath_ng
import numpy as np
import pyarrow as pa
from datasets import Audio, Dataset, DatasetDict, Features, Value, concatenate_datasets
from datasets.fingerprint import Hasher
from datasets.table import InMemoryTable
from jsonpath_ng import JSONPath

from larry.common.config.data.preprocessing import pipeline_config
from larry.common.config.data.preprocessing.preprocessor_configs import (
    ColumnCastingPreprocessorConfig,
    ColumnRenamingPreprocessorConfig,
    NestedExtractionPreprocessorConfig,
    PreprocessorConfig,
    RemoveColumnsPreprocessorConfig,
    RowIndexAssigningPreprocessorConfig,
)
from larry.common.data.utils import features_of
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Preprocessor[C: PreprocessorConfig](Registrable, ABC, root=True):
    """Super class for all step implementations for dataset preprocessing."""

    REGISTRY: ClassVar[TypeRegistry[Preprocessor]]

    _config: C

    def __init__(self, provenance: pipeline_config.Provenance, config: C) -> None:
        """Sets up all subclasses and calls validate() and mutate_provenance() prior to the preprocessing run."""
        self.log = logging.getLogger(type(self).__name__)

        self.provenance = provenance
        self._config = config
        self.validate()
        self.mutate_provenance()

    def __call__(self, batch: dict) -> dict:
        """A warning."""
        raise NotImplementedError("don't do this")

    @property
    def config(self) -> C:
        """Immutable type-hint property return for the config reference."""
        return self._config

    def mutate_provenance(self) -> None:
        """Subclasses can override to add things to the ``Provenance`` instance for the entire run."""
        ...

    def provenance_string(self, batch: dict[str, list[Any]], index: int) -> str:
        """Helper for printing the unique ID for an example within a batch."""
        column_values = {}
        for c in self.provenance.id_columns:
            column_values[c] = batch[c][index]
        strs = []
        for c, v in column_values.items():
            strs.append(f"{c}={v}")
        return f"{{{', '.join(strs)}}}"

    @abstractmethod
    def validate(self) -> None:
        """Overridden to throw exceptions when the current configuration is invalid prior to the preprocessing run."""
        ...

    @abstractmethod
    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """The main entrypoint for subclasses to receive the dataset being preprocessed."""
        ...


class RemoveColumnsPreprocessor(Preprocessor[RemoveColumnsPreprocessorConfig]):
    """A simple preprocessor for removing one or more columns from the dataset being preprocessed."""

    def validate(self) -> None:
        """Prevents a lack of ``config.input_column`` and ``config.output_column`` from raising in the supermethod."""
        pass

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """Simply delegates to ``dataset.remove_columns``, passing the configured columns to remove."""
        return dataset.remove_columns(self.config.remove_columns)


class NestedExtractionPreprocessor(Preprocessor[NestedExtractionPreprocessorConfig]):
    """Takes a JSON text column and produces the results of one or more JSON queries to corresponding output columns."""

    def __init__(self, provenance: pipeline_config.Provenance, config: NestedExtractionPreprocessorConfig) -> None:
        """Initializes the preprocessor with parsed jsonpath queries for extracting the output columns' values."""
        super().__init__(provenance, config)
        self.parsed_queries = {
            mapping["outputColumn"]: jsonpath_ng.parse(mapping["jsonPath"])
            for mapping in self.config.json_path_to_output_column_mappings
        }
        self.output_columns = list(self.parsed_queries.keys())
        self.remove_columns = [*self.config.remove_columns, self.config.input_column]

    def validate(self) -> None:
        """Requires a specified input column and one or more configured mapping queries."""
        if not self.config.input_column:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")

        if not self.config.json_path_to_output_column_mappings:
            raise ValueError(f"json_path_to_output_column_mappings must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """Similar to a mapper but with the output columns unable to be effectively configured prior to runtime."""
        return dataset.map(
            self.preprocess_batch,
            batched=True,
            batch_size=self.config.batch_size,
            writer_batch_size=self.config.writer_batch_size,
            num_proc=self.config.num_proc,
            remove_columns=self.remove_columns,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
            features=self.preprocessing_features(dataset),
            desc=f"{type(self).__name__}: "
                 f"(input_column={self.config.input_column} -> output_columns={self.output_columns})",
        )

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        """All outputs are effectively strings; use ``ColumnCastingMapper`` to cast to different types afterwards."""
        features = features_of(dataset).copy()
        for mapping in self.config.json_path_to_output_column_mappings:
            features[mapping["outputColumn"]] = Value("string")

        return features

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """For each example in a batch, extract the requested query results and produce a new set of columns."""
        collected = {
            mapping["outputColumn"]: []
            for mapping in self.config.json_path_to_output_column_mappings
        }

        for example in batch[self.config.input_column]:
            if isinstance(example, str):
                example = json.loads(example)

            for output_column, query in self.parsed_queries.items():
                collected[output_column].append(self._match_query(query, example))

        return collected

    @staticmethod
    def _match_query(query: JSONPath, example: str) -> str | None:
        """Helper for using the jsonpath query to find the first match."""
        matches = query.find(example)
        if len(matches) > 1:
            raise ValueError(f"Cannot expand the results of filter: {query}; should return either 0 or 1 records")

        matches = matches[0].value if matches else None
        if isinstance(matches, str) and matches.strip() == "":
            matches = None

        return matches


class ColumnCastingPreprocessor(Preprocessor[ColumnCastingPreprocessorConfig]):
    """Preprocessor for overriding the ``FeatureType`` for a column using datasets.cast_column."""

    def validate(self) -> None:
        """Requires one or more configured column cast mappings"""
        if not self.config.column_casts:
            raise ValueError(f"column_casts must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """For each casting configuration, casts the requested column with special handling for specific requests.

        Can also remove columns if the requests column cast "type" is ``None`` (in yaml, nulls are "~").
        """
        for k, v in self.config.column_casts.items():
            if v == "audio_no_decode":
                dataset = dataset.cast_column(k, Audio(decode=False))
            elif v == "audio_decode":
                dataset = dataset.cast_column(k, Audio(decode=True))
            elif v is None:
                dataset = dataset.remove_columns(k)
            else:
                dataset = dataset.cast_column(k, Value(v))

        return dataset


class ColumnRenamingPreprocessor(Preprocessor[ColumnRenamingPreprocessorConfig]):
    """Preprocessor for renaming a column."""

    def validate(self) -> None:
        """Requires one or more column name mappings."""
        if not self.config.column_name_mappings:
            raise ValueError(f"column_name_mappings must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """For each column name mapping, use ``dataset.rename_column`` to rename.

        Supports removing a column if the target name is ``None`` (in yaml, nulls are "~").
        """
        for k, v in self.config.column_name_mappings.items():
            dataset = dataset.remove_columns(k) if v is None else dataset.rename_column(k, v)

        return dataset


class RowIndexAssigningPreprocessor(Preprocessor[RowIndexAssigningPreprocessorConfig]):
    """Preprocessor for assigning a unique, incrementing, non-zero integer row-id to each row in the dataset."""

    def validate(self) -> None:
        """Requires an output_column to produce incrementing unique integer row IDs into."""
        if not self.config.output_column:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """Does the row assignment per-split."""
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                s: self._assign_rows(ds)
                for s, ds in dataset.items()
            })

        return self._assign_rows(dataset)

    def _assign_rows(self, dataset: Dataset) -> Dataset:
        """Technically creates a new dataset that maps each row to its integer index, then concatenates.

        This avoids copying all data in the dataset over for a simple column addition, similar to mappers.
        """
        # noinspection PyProtectedMember
        if dataset._indices is not None:
            dataset = dataset.flatten_indices(writer_batch_size=self.config.writer_batch_size)

        lengths = [len(chunk) for chunk in dataset.data.column(0).chunks]

        # get chunk slice start indices
        starts = np.cumsum([0, *lengths[:-1]])

        # take chunks out of the arange to produce sets of arrays spanning the unique integer ID ranges for each chunk
        # of the dataset (instead of assigning to each in one big go, which requires copying the whole data for workers)
        column = pa.chunked_array(
            [pa.array(np.arange(s, s + n, dtype=np.int64)) for s, n in zip(starts, lengths, strict=True)],
            type=pa.int64(),
        )

        # concatenates the new per-chunk integer IDs to the original dataset, adding the column without requiring a full
        # read and allowing a write of only the parts of each chunk containing the IDs
        # noinspection PyProtectedMember
        rows = Dataset(
            InMemoryTable(pa.Table.from_arrays([column], names=[self.config.output_column])),
            fingerprint=Hasher.hash((dataset._fingerprint, self.config.output_column)),
        )
        return concatenate_datasets([dataset, rows], axis=1)
