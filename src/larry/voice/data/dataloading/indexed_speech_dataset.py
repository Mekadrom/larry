import bisect
from typing import Any, cast

import pyarrow as pa
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
        source_idx = bisect.bisect_right(self.offsets, k) - 1
        config = self.configs[source_idx]
        entry = self.indexes[source_idx][k - self.offsets[source_idx]]

        source = self.sources[source_idx]
        if source is None:
            audio = entry[config.audio_column]
        else:
            row = source[entry["row"]]
            if row["id"] != entry["id"]:
                raise KeyError(f"{config.name} row {entry['row']} is {row['id']}, index expects {entry['id']}")
            audio = row[config.audio_column]

        waveform = bytes_to_waveforms(audio["bytes"], self.sample_rate)[0]
        return {
            "waveform": waveform,
            "text": entry[config.text_column],
            "source": source_idx,
        }

    def source_ids_and_durations(self) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = []
        durations = []
        for s, (config, index) in enumerate(zip(self.configs, self.indexes)):
            table = cast(pa.Table, index.select_columns([config.duration_column]).with_format("arrow")[:])
            column = table.column(config.duration_column).to_numpy()
            durations.append(torch.from_numpy(column).double())
            source_ids.append(torch.full((len(column),), s, dtype=torch.long))
        return torch.cat(source_ids), torch.cat(durations)
