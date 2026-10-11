from functools import partial
from typing import Any, assert_never, cast

from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.pruner_configs import ColumnValuesPrunerConfig, PrunerConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Pruner[I, C: PrunerConfig = PrunerConfig](Preprocessor[C]):
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
            row = cast(I, dict(zip(names, values)))
            keep.append(self._keep_example(row))
        return keep

    def _keep_example(self, example: I) -> bool:
        return (example is None and self.config.prune_nulls) or self.keep_example(example)

    def keep_example(self, example: I) -> bool:
        ...


class ColumnValuesPruner(Pruner[Any, ColumnValuesPrunerConfig]):
    def __init__(self, provenance_columns: list[str], config: ColumnValuesPrunerConfig):
        super().__init__(provenance_columns, config)
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

    def keep_example(self, example: Any) -> bool:
        return self.op(example)

    @classmethod
    def op_matches(cls, op_config: dict[str, Any], example: str) -> bool:
        containing: bool = op_config.get("containing", False)
        case_sensitive: bool = op_config.get("case_sensitive", True)
        strip: bool = op_config.get("strip", True)
        values: list[Any | None] = op_config.get("values", [])

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
    def op_nmatches(cls, op_config: dict[str, Any], example: Any) -> bool:
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
