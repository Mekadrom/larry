from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset, Features, DatasetDict
from datasets.features.features import FeatureType

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, ValueOverrideMapperConfig, \
    MapperConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Mapper[C: MapperConfig = MapperConfig](Preprocessor[C], ABC):
    def __init__(self, config: C) -> None:
        super().__init__(config)
        self.remove_columns = self.config.remove_columns

    def map_kwargs(self) -> dict[str, Any]:
        return dict(
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            remove_columns=self.remove_columns,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
        )

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        return dataset.map(
            self.preprocess_batch,
            features=self.preprocessing_features(dataset),
            **self.map_kwargs()
        )

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return None


class SingleColumnMapper[I, O, C: SingleColumnMapperConfig = SingleColumnMapperConfig](Mapper[C], ABC):
    def __init__(self, config: C) -> None:
        super().__init__(config)
        self.remove_columns = self.config.remove_columns + (
            [self.config.input_column] if self.config.input_column != self.config.output_column else []
        )

    def map_kwargs(self) -> dict[str, Any]:
        k = super().map_kwargs()
        k.update(dict(
            desc=f"{type(self).__name__}: "
                 f"(input_column={self.config.input_column} -> output_column={self.config.output_column})",
        ))
        return k

    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if self.config.output_column is None:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        dataset = super().preprocess_dataset(dataset)
        if self.config.filter_null_outputs:
            dataset = self.filter_nulls(dataset)
        return dataset

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {self.config.output_column: [
            self.preprocess_example(row)
            for row in batch[self.config.input_column]
        ]}

    def preprocess_example(self, example: I) -> O:
        ...

    def filter_nulls(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        return dataset.filter(
            lambda batch: [x is not None for x in batch],
            input_columns=self.config.output_column,
            batched=True,
            batch_size=self.config.batch_size,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
        )

    def output_feature(self) -> FeatureType | None:
        return None


class ValueOverrideMapper(Mapper[ValueOverrideMapperConfig]):
    def validate(self) -> None:
        ...

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output_batch = {}

        def current(column: str) -> list[Any]:
            return output_batch.get(column, batch[column])

        for rule in self.config.mappings:
            source = current(rule.input_column)
            target = output_batch.setdefault(rule.output_column, list(batch[rule.output_column]))
            for i, value in enumerate(source):
                if value == rule.input_column_value:
                    target[i] = rule.output_column_value
        return output_batch
