import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import get_worker_info
from torchaudio import functional as AF

from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.voice.config.data.voice_data_configs import ASRSourceConfig, SpeechSourceConfig, TTSSourceConfig
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from larry.voice.utils import bytes_to_waveforms


class IndexedSpeechDataset[C: SpeechSourceConfig = SpeechSourceConfig](IndexedDataset[C], ABC):
    """Specific dataset type for datasets that are represented by a large materialized form (source) and an index."""

    def __init__(
            self,
            configs: list[C],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            sample_rate: int,
    ) -> None:
        """Initializes index offsets."""
        if not sources:
            raise ValueError("sources are required")

        super().__init__(configs, indexes, sources)
        self.sample_rate = sample_rate

    def get_data(self, source_idx: int, entry: dict[Any, Any]) -> Any:
        config = self.configs[source_idx]
        return entry[config.audio_column]

    def source_ids_and_durations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides new columns for the source dataset ID and duration for each example."""
        source_ids = []
        durations = []
        for s, config in enumerate(self.configs):
            column = self.column(s, config.duration_column).to_numpy()
            durations.append(torch.from_numpy(column).double())
            source_ids.append(torch.full((len(column),), s, dtype=torch.long))

        return torch.cat(source_ids), torch.cat(durations)

    def speaker_keys(self) -> list[str]:
        """Provides the ``SpeakerMapping`` key of every example, read from the indexes without decoding audio."""
        keys: list[str] = []
        for s, (config, index) in enumerate(zip(self.configs, self.indexes, strict=True)):
            if config.speaker_id_column is None:
                keys.extend([SpeakerMapping.key(config, None)] * len(index))
                continue

            keys.extend(SpeakerMapping.key(config, i) for i in self.column(s, config.speaker_id_column).to_pylist())

        return keys

    @abstractmethod
    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        ...


class IndexedASRDataset(IndexedSpeechDataset[ASRSourceConfig]):
    def __init__(
            self,
            configs: list[ASRSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            sample_rate: int,
            speed_factors: tuple[float, ...] = (1.0,),
            seed: int = 0,
            process_index: int = 0,
    ) -> None:
        super().__init__(configs, indexes, sources, sample_rate)
        self.speed_factors = speed_factors
        self.seed = seed
        self.process_index = process_index
        self._speed_rng: tuple[int, random.Random] | None = None

    def speed_rng(self) -> random.Random:
        """A generator for speed factors unique to (seed, process, dataloader worker), independent of global RNGs.

        Rebuilt when the dataset is first used in a new worker, so forked workers never share one copied state.
        """
        worker = get_worker_info()
        worker_key = 0 if worker is None else worker.id + 1
        if self._speed_rng is None or self._speed_rng[0] != worker_key:
            entropy = [self.seed, self.process_index, worker_key]
            self._speed_rng = worker_key, random.Random(int(np.random.SeedSequence(entropy).generate_state(1)[0]))

        return self._speed_rng[1]

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        config = self.configs[source_idx]

        waveform = bytes_to_waveforms(data["bytes"], self.sample_rate)[0]
        factor = self.speed_rng().choice(self.speed_factors)
        if factor != 1.0:
            waveform = AF.resample(waveform, int(self.sample_rate * factor), self.sample_rate)

        return {
            "waveform": waveform,
            "text": entry[config.text_column],
            "source": source_idx,
        }


class IndexedTTSDataset(IndexedSpeechDataset[TTSSourceConfig]):
    def __init__(
            self,
            configs: list[TTSSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            sample_rate: int,
            speaker_mapping: SpeakerMapping,
    ) -> None:
        super().__init__(configs, indexes, sources, sample_rate)
        self.speaker_mapping = speaker_mapping

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        col = self.configs[source_idx].speaker_id_column
        speaker_id_raw = entry[col] if col is not None else None
        speaker_id = self.speaker_mapping[SpeakerMapping.key(self.configs[source_idx], speaker_id_raw)]
        return {
            "waveform": bytes_to_waveforms(data["bytes"], self.sample_rate)[0],
            "unit_ids": entry[self.configs[source_idx].unit_ids_column],
            "speaker_id": speaker_id,
            "source": source_idx,
        }
