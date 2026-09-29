from abc import ABC
from typing import Sequence, Any

from datasets import Dataset, Features
from datasets.features.features import FeatureType

from larry.common.config.mapper_configs import MapperConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Mapper[I, O, C=MapperConfig](Preprocessor[C], ABC):
    def __init__(self, config: C) -> None:
        super().__init__(config)
        self.remove_columns = self.config.remove_columns + (
            [self.config.input_column] if self.config.input_column != self.config.output_column else []
        )

    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if self.config.output_column is None:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    @property
    def map_kwargs(self) -> dict[str, Any]:
        return dict(
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            remove_columns=self.remove_columns,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
            desc=f"{type(self).__name__}: "
                 f"(input_column={self.config.input_column} -> output_column={self.config.output_column})",
        )

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            self.preprocess_batch,
            features=self.preprocessing_features(dataset),
            **self.map_kwargs
        )
        if self.config.filter_null_outputs:
            dataset = self.filter_nulls(dataset)
        return dataset

    def filter_nulls(self, dataset: Dataset) -> Dataset:
        return dataset.filter(
            lambda batch: [x is not None for x in batch],
            input_columns=self.config.output_column,
            batched=True,
            batch_size=self.config.batch_size,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
        )

    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        output_feature = self.output_feature()
        if output_feature is None:
            return None

        features = dataset.features.copy()
        for col in self.remove_columns:
            features.pop(col, None)
        features[self.config.output_column] = output_feature
        return features

    def preprocess_batch(self, batch: dict[str, Sequence[Any]]) -> dict[str, Sequence[Any]]:
        return {self.config.output_column: [self.preprocess_example(row) for row in batch[self.config.input_column]]}

    def preprocess_example(self, example: I) -> O:
        ...

    def output_feature(self) -> FeatureType | None:
        return None
