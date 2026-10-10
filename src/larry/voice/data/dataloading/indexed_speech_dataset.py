import bisect
from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from larry.voice.data.dataloading.voice_dataloading import SpeechSourceConfig
from larry.voice.utils import bytes_to_waveforms


class IndexedSpeechDataset(Dataset[dict[str, Any]]):
    def __init__(
            self,
            configs: list[SpeechSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            sample_rate: int
    ) -> None:
        if not sources:
            raise ValueError("sources are required")

        self.configs = configs
        self.indexes = indexes
        self.sources = sources
        self.sample_rate = sample_rate

        self.offsets = [0]
        for index in indexes:
            self.offsets.append(self.offsets[-1] + len(index))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, k: int) -> dict[str, Any]:
        s = bisect.bisect_right(self.offsets, k) - 1
        config = self.configs[s]
        entry = self.indexes[s][k - self.offsets[s]]

        source = self.sources[s]
        if source is None:
            audio = entry[config.audio_column]
        else:
            row = source[entry["row"]]
            if row["id"] != entry["id"]:
                raise KeyError(f"{config.name} row {entry['row']} is {row['id']}, index expects {entry['id']}")
            audio = row[config.audio_column]

        waveform = bytes_to_waveforms(audio["bytes"], self.sample_rate)[0]
        return {"waveform": waveform, "text": entry[config.text_column]}

    def source_ids_and_durations(self) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = []
        durations = []
        for s, (config, index) in enumerate(zip(self.configs, self.indexes)):
            column = index.with_format("numpy")[config.duration_column]
            durations.append(torch.from_numpy(column).double())
            source_ids.append(torch.full((len(column),), s, dtype=torch.long))
        return torch.cat(source_ids), torch.cat(durations)
