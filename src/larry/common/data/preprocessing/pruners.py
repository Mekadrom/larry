from functools import partial
from typing import Callable, Any

from datasets import Dataset

from larry.common.config.pruner_configs import ColumnValuesPrunerConfig, PrunerConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Pruner[I, C=PrunerConfig](Preprocessor[C]):
    def validate(self) -> None:
        if self.config.input_column is None:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.filter(
            lambda batch: [not p for p in self.prune_batched(batch)],
            input_columns=self.config.input_column,
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
            desc=f"{type(self).__name__}: (input_column={self.config.input_column})",
        )
        return dataset

    def prune_batched(self, batch: list[Any]) -> list[bool]:
        results = []
        for example in batch:
            results.append(self._prune_example(example))
        return results

    def _prune_example(self, example: I) -> bool:
        return (example is None and self.config.prune_nulls) or self.prune_example(example)

    def prune_example(self, example: I) -> bool:
        ...


class ColumnValuesPruner(Pruner[Any, ColumnValuesPrunerConfig]):
    def __init__(self, config: ColumnValuesPrunerConfig):
        super().__init__(config)
        op_dict: dict[str, Callable[..., bool]] = {
            "matches": ColumnValuesPruner.op_matches,
            "nmatches": ColumnValuesPruner.op_nmatches,
        }
        self.op = partial(op_dict.get(self.config.op), self.config.op_config, self.config.values)

    def prune_example(self, example: str) -> bool:
        return self.op(example)

    @classmethod
    def op_matches(cls, op_config: dict[str, Any], values: list[Any | None], example: str) -> bool:
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
    def op_nmatches(cls, op_config: dict[str, Any], values: list[Any | None], example: Any) -> bool:
        return not cls.op_matches(op_config, values, example)
