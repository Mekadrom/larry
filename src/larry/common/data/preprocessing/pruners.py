from functools import partial
from typing import Any, assert_never

from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.data.preprocessing.pruner_configs import ColumnValuesPrunerConfig, PrunerConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Pruner[C: PrunerConfig = PrunerConfig](Preprocessor[C]):
    def validate(self) -> None:
        if not self.config.input_columns:
            raise ValueError(f"input_columns must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        dataset = dataset.filter(
            self.keep_batched,
            input_columns=self.config.input_columns,
            batched=True,
            batch_size=self.config.batch_size,
            num_proc=self.config.num_proc,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
            desc=f"{type(self).__name__}: (input_columns={self.config.input_columns})",
        )
        return dataset

    def keep_batched(self, *columns: list[Any]) -> list[bool]:
        names = self.config.input_columns
        keep = []
        for values in zip(*columns):
            row = dict(zip(names, values))
            keep.append(self._keep_example(row))
        return keep

    def _keep_example(self, example: dict[str, Any]) -> bool:
        has_null = any(
            v is None
            for v in example.values()
        )
        if has_null:
            return not self.config.prune_nulls
        return self.keep_example(example)

    def keep_example(self, example: dict[str, Any]) -> bool:
        ...


class ColumnValuesPruner(Pruner[ColumnValuesPrunerConfig]):
    def __init__(self, provenance: Provenance, config: ColumnValuesPrunerConfig):
        super().__init__(provenance, config)
        op_name = self.config.op
        match op_name:
            case "matches":
                op = ColumnValuesPruner.op_matches
            case "nmatches":
                op = ColumnValuesPruner.op_nmatches
            case "in_range":
                op = ColumnValuesPruner.op_in_range
            case "nin_range":
                op = ColumnValuesPruner.op_nin_range
            case _:
                assert_never(op_name)
        self.op = partial(op, self.config.op_config)

    def keep_example(self, example: dict[str, Any]) -> bool:
        results = [self.op(v) for v in example.values()]
        if self.config.multiple_inputs_behavior == "all":
            return all(results)
        return any(results)

    @classmethod
    def op_matches(cls, op_config: dict[str, Any], example: Any) -> bool:
        if example is None:
            return False
        containing: bool = op_config.get("containing", False)
        case_sensitive: bool = op_config.get("case_sensitive", True)
        strip: bool = op_config.get("strip", True)
        values: list[Any | None] = op_config.get("values", [])

        example_str = str(example)

        if strip:
            example_str = example_str.strip()

        if not case_sensitive:
            example_str = example_str.lower()

        for allowed_value in values:
            allowed_str = str(allowed_value)

            if strip:
                allowed_str = allowed_str.strip()

            if not case_sensitive:
                allowed_str = allowed_str.lower()

            if example_str == allowed_str or (allowed_str is not None and (containing and allowed_str in example_str)):
                return True

        return False

    @classmethod
    def op_nmatches(cls, op_config: dict[str, Any], example: Any) -> bool:
        if example is None:
            return False
        return not cls.op_matches(op_config, example)

    @classmethod
    def op_in_range(cls, op_config: dict[str, Any], example: Any) -> bool:
        if example is None:
            return False

        min_value: float | None = op_config.get("min_value", None)
        max_value: float | None = op_config.get("max_value", None)

        if min_value is not None:
            if example <= min_value:
                return False
        if max_value is not None:
            if example >= max_value:
                return False
        return True

    @classmethod
    def op_nin_range(cls, op_config: dict[str, Any], example: Any) -> bool:
        if example is None:
            return False
        return not cls.op_in_range(op_config, example)
