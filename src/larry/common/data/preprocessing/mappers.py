import hashlib
import os.path
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import torch.cuda
from datasets import Dataset, Features, DatasetDict, concatenate_datasets, Value
from datasets.features.features import FeatureType

from larry.common.config.data.preprocessing.mapper_configs import SingleColumnMapperConfig, ValueOverrideMapperConfig, \
    MapperConfig, UrlMapperConfig, UrlBytesMapperConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.preprocessors import Preprocessor


class Mapper[C: MapperConfig = MapperConfig](Preprocessor[C], ABC):
    def __init__(self, provenance: Provenance, config: C) -> None:
        super().__init__(provenance, config)
        if self.config.remove_columns:
            self.remove_columns = [c for c in self.config.remove_columns if c is not None]
        else:
            self.remove_columns = []
        self.input_columns = [c for c in self.config.input_columns if c is not None]
        self.output_columns = [c for c in self.config.output_columns if c is not None]
        self.device_index = 0

    def validate(self) -> None:
        if not self.config.input_columns:
            raise ValueError(f"input_columns must be specified for {self.__class__.__name__}")
        if not self.config.output_columns:
            raise ValueError(f"output_columns must be specified for {self.__class__.__name__}")

    def map_kwargs(self) -> dict[str, Any]:
        return dict(
            batched=True,
            batch_size=self.config.batch_size,
            writer_batch_size=self.config.writer_batch_size,
            num_proc=self.config.num_proc,
            remove_columns=self.remove_columns,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
        )

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict({
                split: self._preprocess_dataset(split_ds)
                for split, split_ds in dataset.items()
            })
        else:
            dataset = self._preprocess_dataset(dataset)
        return self._filter_nulls(dataset)

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        # noinspection PyProtectedMember
        if dataset._indices is not None:
            dataset = dataset.flatten_indices(
                num_proc=self.config.num_proc,
                writer_batch_size=self.config.writer_batch_size
            )
        out = dataset.map(
            self._preprocess_columns,
            input_columns=self.input_columns,
            features=self.output_features(dataset),
            with_rank=True,
            **{**self.map_kwargs(), "remove_columns": dataset.column_names},
        )
        drop = [c for c in dataset.column_names if c in out.column_names or c in self.remove_columns]
        return concatenate_datasets([dataset.remove_columns(drop), out], axis=1)

    def _preprocess_columns(self, *args: Any) -> dict[str, list[Any]]:
        *columns, rank = args
        return self._preprocess_batch(dict(zip(self.input_columns, columns)), rank)

    def _preprocess_batch(self, batch: dict[str, list[Any]], rank: int | None) -> dict[str, list[Any]]:
        if rank is not None:
            n_devices = torch.cuda.device_count()
            if n_devices > 0:
                self.device_index = rank % n_devices
        return self.preprocess_batch(batch)

    @abstractmethod
    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...

    def _filter_nulls(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, Dataset):
            dataset = self.filter_nulls(dataset)
        elif isinstance(dataset, DatasetDict):
            for split, split_ds in dataset.items():
                dataset[split] = self.filter_nulls(split_ds)
        return dataset

    def filter_nulls(self, dataset: Dataset) -> Dataset:
        n = sum(dataset.data.column(c).null_count for c in self.output_columns)
        if self.config.filter_null_outputs and n > 0:
            return dataset.filter(
                lambda *cols: [all(v is not None for v in row) for row in zip(*cols)],
                input_columns=self.output_columns,
                batched=True,
                batch_size=self.config.batch_size,
                writer_batch_size=self.config.writer_batch_size,
                keep_in_memory=not self.config.cache_results,
                load_from_cache_file=self.config.cache_results,
            )
        return dataset

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return None


class SingleColumnMapper[I, O, C: SingleColumnMapperConfig = SingleColumnMapperConfig](Mapper[C], ABC):
    def validate(self) -> None:
        super().validate()
        if not self.config.input_column:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if not self.config.output_column:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        column_in = batch[self.config.input_column]
        column_out = []
        for example in column_in:
            column_out.append(self.preprocess_example(example))
        return {self.config.output_column: column_out}

    def preprocess_example(self, example: I) -> O:
        ...

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        output_feature = self.output_feature()
        if output_feature is None:
            return None
        return Features({self.config.output_column: output_feature})

    @abstractmethod
    def output_feature(self) -> FeatureType | None:
        ...


class ValueOverrideMapper(Mapper[ValueOverrideMapperConfig]):
    def validate(self) -> None:
        super().validate()
        if not self.config.mappings:
            raise ValueError(f"mappings_config must be specified for {self.__class__.__name__}")

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


_SESSIONS: dict[str, requests.Session] = {}


class UrlMapper[O, C: UrlMapperConfig = UrlMapperConfig](SingleColumnMapper[str, O | None, C], ABC):
    def __init__(self, provenance: Provenance, config: C) -> None:
        super().__init__(provenance, config)
        if self.config.download_cache_dir is not None:
            self.download_cache_dir = Path(self.config.download_cache_dir).expanduser().resolve()
            os.makedirs(self.download_cache_dir, exist_ok=True)
        else:
            self.download_cache_dir = None

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        column_in = batch[self.config.input_column]
        column_out = []
        for example in column_in:
            column_out.append(self.preprocess_example(example))
        return {self.config.output_column: column_out}

    def preprocess_example(self, example: str) -> O | None:
        url_hash = str(hashlib.sha256(example.encode('utf-8')).hexdigest())
        if self.download_cache_dir is not None:
            cache_file_path = self.download_cache_dir / url_hash
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
            # lazy init session
            session = self._lazy_get_session()
            if session is None:
                raise ValueError("session was not initialized")

            for attempt in range(self.config.max_retries + 1):
                sleep = self.config.busy_wait + (self.config.retry_backoff * 2 ** (attempt - 1) if attempt else 0)
                time.sleep(sleep)
                try:
                    response = session.get(example, timeout=self.config.timeout)
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
            f"this row will contain a null for one of output_columns={self.config.output_columns}, "
            f"and nulls {'are' if self.config.filter_null_outputs else 'are not'} filtered out after mapping"
        )
        return None

    def _lazy_get_session(self) -> requests.Session:
        key = repr(self.config)
        if key not in _SESSIONS:
            session = requests.Session()
            session.headers["User-Agent"] = self.user_agent()
            _SESSIONS[key] = session
        return _SESSIONS[key]

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
    def user_agent(self) -> str:
        ...

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
