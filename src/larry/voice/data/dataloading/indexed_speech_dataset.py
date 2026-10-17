import bisect
import random
from typing import Any, cast

import pyarrow as pa
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torchaudio import functional as AF

from larry.voice.config.preprocessing.voice_configs import SpeechSourceConfig
from larry.voice.data.preprocessing.global_speaker_map import SpeakerMapping
from larry.voice.utils import bytes_to_waveforms


class IndexedSpeechDataset(Dataset[dict[str, Any]]):
    """Specific dataset type for datasets that are represented by a large materialized form (source) and an index."""

    def __init__(
            self,
            configs: list[SpeechSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            sample_rate: int,
            speed_factors: tuple[float, ...] = (1.0,),
    ) -> None:
        """Initializes index offsets."""
        if not sources:
            raise ValueError("sources are required")

        self.configs = configs
        self.indexes = indexes
        self.sources = sources
        self.sample_rate = sample_rate
        self.speed_factors = speed_factors

        self.offsets = [0]
        for index in indexes:
            self.offsets.append(self.offsets[-1] + len(index))

    def __len__(self) -> int:
        """Returns the final offset index as the length."""
        return self.offsets[-1]

    def __getitem__(self, k: int) -> dict[str, Any]:
        """Returns a single item as a dictionary from the requested index, determining which source it falls in."""
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
        factor = random.choice(self.speed_factors)
        if factor != 1.0:
            waveform = AF.resample(waveform, int(self.sample_rate * factor), self.sample_rate)

        return {
            "waveform": waveform,
            "text": entry[config.text_column],
            "source": source_idx,
        }

    def source_ids_and_durations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides new columns for the source dataset ID and duration for each example."""
        source_ids = []
        durations = []
        for s, (config, index) in enumerate(zip(self.configs, self.indexes, strict=False)):
            table = cast(pa.Table, index.select_columns([config.duration_column]).with_format("arrow")[:])
            column = table.column(config.duration_column).to_numpy()
            durations.append(torch.from_numpy(column).double())
            source_ids.append(torch.full((len(column),), s, dtype=torch.long))

        return torch.cat(source_ids), torch.cat(durations)

    def speaker_keys(self) -> list[str]:
        """Provides the ``SpeakerMapping`` key of every example, read from the indexes without decoding audio."""
        keys: list[str] = []
        for config, index in zip(self.configs, self.indexes, strict=True):
            if config.speaker_id_column is None:
                keys.extend([SpeakerMapping.key(config, None)] * len(index))
                continue

            table = cast(pa.Table, index.select_columns([config.speaker_id_column]).with_format("arrow")[:])
            keys.extend(SpeakerMapping.key(config, s) for s in table.column(config.speaker_id_column).to_pylist())

        return keys
