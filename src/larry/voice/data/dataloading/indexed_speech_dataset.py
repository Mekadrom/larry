from typing import Any

import torch
from datasets import Dataset as HFDataset
from torchaudio import functional as AF

from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.common.data.dataloading.worker_random import WorkerRandom
from larry.voice.config.data.voice_data_configs import ASRSourceConfig, SpeechSourceConfig, TTSSourceConfig
from larry.voice.data.preprocessing.speaker_mapping import SpeakerMapping
from larry.voice.utils import bytes_to_waveforms


class IndexedSpeechDataset[C: SpeechSourceConfig = SpeechSourceConfig](IndexedDataset[C]):
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

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        return {
            "waveform": bytes_to_waveforms(data["bytes"], self.sample_rate)[0],
            "source": source_idx,
        }


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
        self.random = WorkerRandom(seed, process_index)

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        config = self.configs[source_idx]

        waveform = bytes_to_waveforms(data["bytes"], self.sample_rate)[0]
        factor = self.random.get().choice(self.speed_factors)
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
