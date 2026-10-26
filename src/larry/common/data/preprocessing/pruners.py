import os
from functools import partial
from typing import Any, assert_never

from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.config.data.preprocessing.pruner_configs import (
    ColumnValuesPrunerConfig,
    DoNotTrainPrunerConfig,
    MatchOpConfig,
    PrunerConfig,
    RangeOpConfig,
)
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Pruner[C: PrunerConfig = PrunerConfig](Preprocessor[C]):
    """Superclass for all pruners, doesn't require overrides but doesn't do anything by itself."""

    def validate(self) -> None:
        """Pruners don't produce outputs."""
        if not self.config.input_columns:
            raise ValueError(f"input_columns must be specified for {self.__class__.__name__}")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """Uses ``dataset.filter`` instead of ``dataset.map``.

        This provides the batch function (``self.keep_batched``) with an args list of column data, rather than a dict.
        """
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
        """For each example in a batch, converts the list of column lists into a dict and returns a boolean for each.

        True means the example should be kept, False mean the example should be pruned.
        """
        names = self.config.input_columns
        keep = []
        for values in zip(*columns, strict=True):
            row = dict(zip(names, values, strict=True))
            keep.append(self._keep_example(row))
        return keep

    def _keep_example(self, example: dict[str, Any]) -> bool:
        """Internal keep_example entrypoint that just adds that has_null check for any input column into the ``Pruner``.

        This check runs prior to any other overridden ``Pruner`` behavior.
        """
        has_null = any(
            v is None
            for v in example.values()
        )
        if has_null:
            return not self.config.prune_nulls
        return self.keep_example(example)

    def keep_example(self, example: dict[str, Any]) -> bool:
        """Default entrypoint for subclasses of ``Pruner``, takes a single example dict and returns a decision."""
        ...


class ColumnValuesPruner(Pruner[ColumnValuesPrunerConfig]):
    """Pruner implementation that does complex logic to determine if a row should be kept based on its values."""

    def __init__(self, provenance: Provenance, config: ColumnValuesPrunerConfig) -> None:
        """Sets up the delegate operation for determining whether to keep or prune rows."""
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
        """Given a single example, runs the operation on all input columns."""
        results = [self.op(v) for v in example.values()]
        if self.config.multiple_inputs_behavior == "all":
            return all(results)

        return any(results)

    @classmethod
    def op_matches(cls, op_config: MatchOpConfig, example: Any) -> bool:
        """Whether this example's value in a single column matches the operation's configured values.

        Other configuration included in ``self.config.op_config`` is used to tune the matching behavior.
        Returns False for nulls, overriding the ``prune_nulls`` behavior.
        """
        if example is None:
            return False

        containing = op_config.containing
        case_sensitive = op_config.case_sensitive
        strip = op_config.strip
        values = op_config.values

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
    def op_nmatches(cls, op_config: MatchOpConfig, example: Any) -> bool:
        """Inverts ``op_matches``, except for null examples which still get pruned."""
        if example is None:
            return False

        return not cls.op_matches(op_config, example)

    @classmethod
    def op_in_range(cls, op_config: RangeOpConfig, example: Any) -> bool:
        """Tests that a number value is within a max and min value range, unbounded if either is null.

        If example is None, returns False in order to prune it, overriding the default ``prune_nulls`` configuration.
        """
        if example is None:
            return False

        min_value = op_config.min_value
        max_value = op_config.max_value

        if min_value is not None and example <= min_value:
            return False
        return not (max_value is not None and example >= max_value)

    @classmethod
    def op_nin_range(cls, op_config: RangeOpConfig, example: Any) -> bool:
        """Inverts ``op_in_range``, except when example is None, which still returns False."""
        if example is None:
            return False
        return not cls.op_in_range(op_config, example)


class DoNotTrainPruner(Pruner[DoNotTrainPrunerConfig]):
    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        self.require_api_key()
        return super().preprocess_dataset(dataset)

    def require_api_key(self) -> None:
        if not os.environ.get(self.config.api_key_env_var):
            raise ValueError(
                f"{type(self).__name__} needs a Spawning API key in ${self.config.api_key_env_var}; without one, "
                f"datadiligence treats every url as allowed"
            )

    def allowed_urls(self, urls: list[str]) -> set[str]:
        from datadiligence.rules import SpawningAPI

        if not urls:
            return set()

        api = SpawningAPI(user_agent=self.config.user_agent)
        api.api_key = os.environ[self.config.api_key_env_var]
        return set(api.filter_allowed(urls=urls))

    def keep_batched(self, *columns: list[Any]) -> list[bool]:
        urls = sorted({url for column in columns for url in column if url is not None})
        allowed = self.allowed_urls(urls)
        keep = []
        for values in zip(*columns, strict=True):
            if any(value is None for value in values):
                keep.append(not self.config.prune_nulls and all(v in allowed for v in values if v is not None))
            else:
                keep.append(all(value in allowed for value in values))

        pruned = keep.count(False)
        if pruned:
            self.log.info(f"pruned {pruned} of {len(keep)} rows opted out of training or without urls")
        return keep
