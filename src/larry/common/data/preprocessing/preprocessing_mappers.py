import abc
import json
import logging
from abc import ABC, abstractmethod
from functools import partial
from itertools import compress
from typing import Any, Callable
from typing import Sequence, Mapping

import jsonpath_ng
from datasets import Dataset, Value
from datasets import Features
from jsonpath_ng import JSONPath

from larry.common.config.preprocessing_mapper_configs import ColumnValuesPruningPreprocessingMapperConfig, \
    PruningPreprocessingMapperConfig
from larry.common.config.preprocessing_mapper_configs import PreprocessingMapperConfig, \
    NestedExtractionPreprocessingMapperConfig


class PreprocessingMapper[I = None, O = None](abc.ABC):
    """Abstract class for a single worker instance that sees only a batch at a time, rather than a whole dataset."""
    config: PreprocessingMapperConfig

    def __init__(self, config: PreprocessingMapperConfig) -> None:
        self.config = config
        self.validate()
        self.remove_columns = self.config.remove_columns + (
            [self.config.input_column] if self.config.input_column != self.config.output_column else []
        )
        self.log = logging.getLogger(type(self).__name__)

    def __call__(self, batch: dict) -> dict:
        """A warning."""
        raise NotImplementedError("don't do this")

    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if self.config.output_column is None:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        self.log.info(f"remove_columns={self.remove_columns}")
        return dataset.map(
            self.preprocess_batch,
            batched=self.config.batched,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            remove_columns=self.remove_columns,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
            features=self.preprocessing_features(dataset)
        )

    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        return None

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, Sequence[Any]]:
        output = {}
        column = batch[self.config.input_column]
        keep_rows = [True for _ in range(len(column))]
        for i, example in enumerate(column):
            if example is None:
                raise ValueError(
                    f"{type(self).__name__}: example {i} in column {self.config.input_column!r} is None")
            preprocessed_example = self.preprocess_example(example)
            if isinstance(self.config.output_column, list) and isinstance(preprocessed_example, (dict, Mapping)):
                for mapping in self.config.output_column:
                    preprocessed_batch = output.get(mapping["value"], [])
                    preprocessed_batch.append(preprocessed_example[mapping["key"]])
                    output[mapping["value"]] = preprocessed_batch
            else:
                if preprocessed_example is None:
                    keep_rows[i] = False
                elif isinstance(preprocessed_example, (dict, Mapping)):
                    preprocessed_example = preprocessed_example[self.config.output_column]
                preprocessed_batch = output.get(self.config.output_column, [])
                output[self.config.output_column] = preprocessed_batch + [preprocessed_example]
        batch.update(output)
        return {
            key: list(compress(values, keep_rows))
            for key, values in batch.items()
            if key not in self.remove_columns
        }

    def preprocess_example(self, example: I) -> O | dict[str, O]:
        raise NotImplementedError


class PruningPreprocessingMapper[I = None](PreprocessingMapper[I], ABC):
    config: PruningPreprocessingMapperConfig

    def __init__(self, config: PruningPreprocessingMapperConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.remove_columns = self.config.remove_columns + (
            [self.config.input_column] if self.config.remove_column else []
        )

    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        self.log.info(f"remove_columns={self.remove_columns}")
        dataset = dataset.filter(
            self.prune_batched,
            batched=self.config.batched,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
        )
        if self.remove_columns:
            dataset = dataset.remove_columns(self.remove_columns)
        return dataset

    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        features = dataset.features.copy()
        for col in self.remove_columns:
            features.pop(col, None)
        return features

    def prune_batched(self, batch: dict[str, list[Any]]) -> list[bool]:
        column = batch[self.config.input_column]
        results = []
        for example in column:
            if example is None and self.config.drop_missing:
                is_pruned = True
            else:
                is_pruned = not self.prune_example(example)
            results.append(is_pruned)
        return results

    @abstractmethod
    def prune_example(self, example: I) -> bool:
        ...


class RemoveColumnsPreprocessingMapper(PreprocessingMapper):
    def __init__(self, config: PreprocessingMapperConfig) -> None:
        super().__init__(config)
        self.config = config

    def validate(self) -> None:
        """Prevents a lack of ``config.input_column`` and ``config.output_column`` from raising in the supermethod."""
        pass

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.remove_columns(self.config.remove_columns)


class NestedExtractionPreprocessingMapper(PreprocessingMapper[str, str]):
    config: NestedExtractionPreprocessingMapperConfig

    def __init__(self, config: NestedExtractionPreprocessingMapperConfig) -> None:
        super().__init__(config=config)
        self.config = config
        self.parsed_queries = {
            mapping["outputColumn"]: jsonpath_ng.parse(mapping["jsonPath"])
            for mapping in self.config.json_path_to_output_column_mappings
        }

    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")

    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        features = dataset.features.copy()
        for col in self.remove_columns:
            features.pop(col, None)
        for mapping in self.config.json_path_to_output_column_mappings:
            features[mapping["outputColumn"]] = Value("string")
        return features

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, Sequence[Any]]:
        collected = {
            mapping["outputColumn"]: []
            for mapping in self.config.json_path_to_output_column_mappings
        }
        column = batch[self.config.input_column]
        n_examples = len(column)

        keep_rows: list[bool] = [True for _ in range(n_examples)]

        for i, example in enumerate(column):
            if isinstance(example, str):
                example = json.loads(example)

            keep_rows[i], matched = self._match_queries(example)
            for k, v in matched.items():
                collected[k] += [v]

        batch.update(collected)

        return {
            key: list(compress(values, keep_rows))
            for key, values in batch.items()
            if key not in self.remove_columns
        }

    def _match_queries(self, example: Any) -> tuple[bool, dict[str, Any]]:
        num_expected_outputs = len(self.parsed_queries)

        keep_row = True
        collected = {}

        num_failed_queries = 0
        for output_column, query in self.parsed_queries.items():
            matched = self._match_query(query, example)
            if matched is None:
                num_failed_queries += 1
            collected[output_column] = matched

        if self.config.drop_missing_type == "all":
            keep_row = num_failed_queries < num_expected_outputs
        elif self.config.drop_missing_type == "any":
            keep_row = num_failed_queries == 0

        return keep_row, collected

    def _match_query(self, query: JSONPath, example: str):
        matches = query.find(example)
        if len(matches) > 1:
            raise ValueError(f"Cannot expand the results of filter: {query}. "
                             f"Make the jsonpath return either 1 or 0 records.")
        matches = matches[0].value if matches else None
        if isinstance(matches, str) and matches.strip() == "":
            matches = None
        return matches


class ColumnValuesPruningPreprocessingMapper(PruningPreprocessingMapper[str]):
    config: ColumnValuesPruningPreprocessingMapperConfig

    def __init__(self, config: ColumnValuesPruningPreprocessingMapperConfig):
        super().__init__(config=config)
        self.config = config
        op_dict: dict[str, Callable[..., bool]] = {
            "in": ColumnValuesPruningPreprocessingMapper.op_in,
            "nin": ColumnValuesPruningPreprocessingMapper.op_nin,
            "eq": ColumnValuesPruningPreprocessingMapper.op_eq,
        }
        self.op = partial(op_dict.get(self.config.op), self.config.op_config, self.config.values)

    def prune_example(self, example: str) -> bool:
        return self.op(example)

    @classmethod
    def op_in(cls, op_config: dict[str, Any], values: list[Any | None], example: str) -> bool:
        containing: bool = op_config.get("containing", False)
        case_sensitive: bool = op_config.get("case_sensitive", True)
        strip: bool = op_config.get("strip", True)

        if strip and isinstance(example, str):
            example = example.strip()

        if not case_sensitive and isinstance(example, str):
            example = str(example).lower()

        for allowed_value in values:
            if isinstance(allowed_value, str) and not case_sensitive:
                allowed_value = str(allowed_value).lower()
            if example == allowed_value or (allowed_value is not None and (containing and allowed_value in example)):
                return True
        return False

    @classmethod
    def op_nin(cls, op_config: dict[str, Any], values: list[Any | None], example: Any) -> bool:
        return not cls.op_in(op_config, values, example)

    @classmethod
    def op_eq(cls, op_config: dict[str, Any], values: list[Any | None], example: Any) -> bool:
        strip: bool = op_config.get("strip", True)
        if strip:
            example = example.strip()
        logging.info(f"values={values}")
        return any(example == value for value in values)