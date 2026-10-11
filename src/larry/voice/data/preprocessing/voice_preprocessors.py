import hashlib
from functools import partial
from typing import Any, cast

import numpy as np
from datasets import Dataset, DatasetDict, NamedSplit

from larry.common.data.preprocessing.preprocessors import Preprocessor
from larry.voice.config.preprocessing.voice_preprocessing_configs import SpeakerValidationSplittingPreprocessorConfig


class SpeakerValidationSplittingPreprocessor(Preprocessor[SpeakerValidationSplittingPreprocessorConfig]):
    def validate(self) -> None:
        if not self.config.mode or not self.config.mode in ["random", "matching"]:
            raise ValueError("mode must specified and one of [random, matching]")
        if self.config.mode == "random" and not self.config.validation_percent:
            raise ValueError(f"random mode requires setting validation_percent for {type(self).__name__}")
        if self.config.mode == "matching" and not self.config.matching_values:
            raise ValueError(f"matching mode requires setting matching_values for {type(self).__name__}")

    def _preprocessing_kwargs(self) -> dict[str, Any]:
        return dict(
            batched=True,
            batch_size=self.config.batch_size,
            writer_batch_size=self.config.writer_batch_size,
            num_proc=self.config.num_proc,
            keep_in_memory=not self.config.cache_results,
            load_from_cache_file=self.config.cache_results,
            input_columns=[self.config.speaker_id_column],
        )

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        others: dict[str | NamedSplit, Dataset] = {}
        if isinstance(dataset, DatasetDict) and self.config.operate_on_split is not None:
            for name, split in dataset.items():
                if name != self.config.operate_on_split:
                    others[name] = split
            dataset = dataset[self.config.operate_on_split]

        if isinstance(dataset, DatasetDict):
            raise ValueError(
                f"dataset is already split; further splitting behavior is undefined, please specify operate_on_split"
            )

        is_train = None
        is_validation = None
        if self.config.mode == "random":
            is_train = self._hashed_is_train
            is_validation = self._hashed_is_validation
        elif self.config.mode == "matching":
            matching_values = self.config.matching_values
            if matching_values:
                is_train = partial(self._speaker_id_nmatches, matching_values)
                is_validation = partial(self._speaker_id_matches, matching_values)

        if is_train is None or is_validation is None:
            raise ValueError(f"unknown mode: {self.config.mode}")

        train_dataset = dataset.filter(
            is_train,
            **self._preprocessing_kwargs()
        )
        validation_dataset = dataset.filter(
            is_validation,
            **self._preprocessing_kwargs()
        )

        return DatasetDict({
            **others,
            "train": train_dataset,
            "validation": validation_dataset,
        })

    def _speaker_bucket(self, speaker_id: str) -> int:
        digest = hashlib.blake2b(speaker_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % 100

    def _hashed_is_validation(self, speaker_ids: list[Any]) -> list[bool]:
        validation_percent = self.config.validation_percent
        if validation_percent is None:
            raise ValueError("validation_percent must be specified for random splitting mode")

        return [
            self._speaker_bucket(str(s)) < validation_percent
            for s in speaker_ids
        ]

    def _hashed_is_train(self, speaker_ids: list[Any]) -> list[bool]:
        return [
            not held_out
            for held_out in self._hashed_is_validation(speaker_ids)
        ]

    def _speaker_id_matches(self, allowed_values: list[str], speaker_ids: list[Any]) -> list[bool]:
        return [
            s in allowed_values
            for s in speaker_ids
        ]

    def _speaker_id_nmatches(self, allowed_values: list[str], speaker_ids: list[Any]) -> list[bool]:
        return [
            not held_out
            for held_out in self._speaker_id_matches(allowed_values, speaker_ids)
        ]

    def _log_split(self, name: str, split: Dataset) -> None:
        n_speakers = len(split.unique(self.config.speaker_id_column))
        hours: float | None = None
        if self.config.duration_column in split.column_names:
            durations = cast(np.ndarray, split.with_format("numpy")[self.config.duration_column])
            hours = float(durations.sum()) / 3600
        self.log.info(f"{name}: rows={len(split)} speakers={n_speakers} hours={hours}")
