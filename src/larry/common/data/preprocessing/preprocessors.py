import json
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any, ClassVar

import jsonpath_ng
from datasets import Dataset, Features, Value, DatasetDict
from jsonpath_ng import JSONPath

from larry.common.config.data.preprocessing.preprocessor_configs import RemoveColumnsPreprocessorConfig, \
    NestedExtractionPreprocessorConfig, PreprocessorConfig
from larry.common.data.utils import features_of
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Preprocessor[C: PreprocessorConfig](Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[Preprocessor]]

    _config: C

    def __init__(self, provenance_columns: list[str], config: C) -> None:
        self.provenance_columns = provenance_columns
        self._config = config
        self.validate()
        self.log = logging.getLogger(type(self).__name__)

    @property
    def config(self) -> C:
        return self._config

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        ...

    def __call__(self, batch: dict) -> dict:
        """A warning."""
        raise NotImplementedError("don't do this")

    def provenance_string(self, batch: dict[str, list[Any]], index: int) -> str:
        column_values = {}
        for c in self.provenance_columns:
            column_values[c] = batch[c][index]
        strs = []
        for c, v in column_values.items():
            strs.append(f"{c}={v}")
        return f"{{{', '.join(strs)}}}"


class RemoveColumnsPreprocessor(Preprocessor[RemoveColumnsPreprocessorConfig]):
    def validate(self) -> None:
        """Prevents a lack of ``config.input_column`` and ``config.output_column`` from raising in the supermethod."""
        pass

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        return dataset.remove_columns(self.config.remove_columns)


class NestedExtractionPreprocessor(Preprocessor[NestedExtractionPreprocessorConfig]):
    def __init__(self, provenance_columns: list[str], config: NestedExtractionPreprocessorConfig) -> None:
        super().__init__(provenance_columns, config)
        self.parsed_queries = {
            mapping["outputColumn"]: jsonpath_ng.parse(mapping["jsonPath"])
            for mapping in self.config.json_path_to_output_column_mappings
        }
        self.output_columns = list(self.parsed_queries.keys())
        self.remove_columns = self.config.remove_columns.append(self.config.input_column)

    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
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
        features = features_of(dataset).copy()
        for mapping in self.config.json_path_to_output_column_mappings:
            features[mapping["outputColumn"]] = Value("string")
        return features

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
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
    def _match_query(query: JSONPath, example: str):
        matches = query.find(example)
        if len(matches) > 1:
            raise ValueError(f"Cannot expand the results of filter: {query}; should return either 0 or 1 records")
        matches = matches[0].value if matches else None
        if isinstance(matches, str) and matches.strip() == "":
            matches = None
        return matches
