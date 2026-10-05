import hashlib
import os.path
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from datasets import Dataset, Features, DatasetDict
from datasets.features.features import FeatureType, Value

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, ValueOverrideMapperConfig, \
    MapperConfig, UrlMapperConfig, UrlBytesMapperConfig
from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.common.data.utils import features_of


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

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        output_feature = self.output_feature()
        if output_feature is None:
            return None

        features = super().preprocessing_features(dataset)
        if features is None:
            features = features_of(dataset).copy()
        for column in self.remove_columns:
            features.pop(column, None)
        features.update({self.config.output_column: output_feature})
        return features

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


class UrlMapper[O, C: UrlMapperConfig = UrlMapperConfig](SingleColumnMapper[str, O | None, C], ABC):
    def __init__(self, config: C) -> None:
        super().__init__(config)
        if self.config.download_cache_dir is not None:
            self.download_cache_dir = Path(self.config.download_cache_dir).expanduser().resolve()
            os.makedirs(self.download_cache_dir, exist_ok=True)
        else:
            self.download_cache_dir = None
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.user_agent()

    def preprocess_example(self, example: str) -> O | None:
        if self.download_cache_dir is not None:
            cache_file_path = self.download_cache_dir / f"{hashlib.sha256(example.encode('utf-8')).hexdigest()}"
        else:
            cache_file_path = None

        content = None
        if cache_file_path is not None and cache_file_path.exists() and os.path.getsize(cache_file_path) > 0:
            self.log.info(f"cache hit for url: {example} at {str(cache_file_path)}")
            content = cache_file_path.read_bytes()
            try:
                self.validate_content(example, content)
            except Exception:
                self.log.error(
                    f"Cached content at {cache_file_path} did not pass validation; redownloading from: {example}"
                )
                content = None

        if content is None:
            for attempt in range(self.config.max_retries + 1):
                sleep = self.config.busy_wait + (self.config.retry_backoff * 2 ** (attempt - 1) if attempt else 0)
                time.sleep(sleep)
                try:
                    response = self.session.get(example, timeout=self.config.timeout)
                except (requests.ConnectionError, requests.Timeout):
                    self.log.warning(f"network error (attempt {attempt + 1}) for {example}", exc_info=True)
                    continue

                if response.status_code == 403:
                    # we have been kicked off the website in the worst way possible: ip ban. abort the mission
                    raise ValueError(
                        f"You have been IP banned from {urlparse(example).hostname} - preprocessing has been cancelled."
                    )
                if response.status_code == 404:
                    # not found - typical, can actually be retried because sometimes this is transient in my experience
                    continue
                if response.status_code == 429:
                    time.sleep(float(response.headers.get("Retry-After", 60)))
                    continue
                if 500 <= response.status_code < 600:
                    continue

                if response.ok:
                    try:
                        self._validate_response(example, response)
                    except Exception:
                        self.log.warning(f"validation error (attempt {attempt + 1}) for {example}", exc_info=True)
                        continue

                    content = response.content

                    if cache_file_path is not None:
                        tmp = cache_file_path.with_suffix(".part")
                        tmp.write_bytes(content)
                        os.replace(tmp, cache_file_path)

                    break
                else:
                    self.log.error(f"{response.status_code} for {example}: {response.text[:200]!r}")

        if content is not None:
            return self.extract_content(example, content)

        self.log.warning(
            f"Failed to obtain content for url={example} - "
            f"this row will contain a null for {self.config.output_column}, "
            f"and nulls {'are' if self.config.filter_null_outputs else 'are not'} filtered out after mapping"
        )
        return None

    @abstractmethod
    def user_agent(self) -> str:
        ...

    def _validate_response(self, example: str, response: requests.Response) -> None:
        if "Content-Length" in response.headers:
            expected = int(response.headers["Content-Length"])
            actual = response.raw.tell()
            if expected != actual:
                raise ValueError(
                    f"Content-Length header value ({expected}) does not match real content length {actual}"
                )
        self.validate_content(example, response.content)

    @abstractmethod
    def validate_content(self, example: str, content: bytes) -> None:
        ...

    @abstractmethod
    def extract_content(self, example: str, content: bytes) -> O:
        ...


class UrlBytesMapper(UrlMapper[bytes, UrlBytesMapperConfig]):
    def user_agent(self) -> str:
        return "larry-bytes-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def validate_content(self, example: str, content: bytes) -> None:
        if self.config.validate_starts_with is not None:
            if not content.startswith(self.config.validate_starts_with.encode()):
                raise ValueError(f"not a pdf: {example} - {str(content)[:200]!r}")

    def extract_content(self, example: str, content: bytes) -> bytes:
        return content

    def output_feature(self) -> FeatureType | None:
        return Value("binary")
