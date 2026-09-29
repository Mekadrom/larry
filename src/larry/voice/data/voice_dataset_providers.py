import csv
from pathlib import Path
from typing import Sequence, Mapping, Any

from torchcodec.decoders import AudioDecoder

from larry.common.data.cached_dataset_providers import MediaCachedDatasetProvider
from larry.voice.config.voice_dataset_configs import CommonVoiceMediaCachedDatasetConfig


class CommonVoiceMediaCachedDatasetProvider(MediaCachedDatasetProvider[CommonVoiceMediaCachedDatasetConfig]):
    _archive_root: Path
    _audio_durations: dict[str, float]

    def __init__(self, config: CommonVoiceMediaCachedDatasetConfig) -> None:
        super().__init__(config)
        self._archive_root = self.dir / CommonVoiceMediaCachedDatasetConfig.ARCHIVE_ROOT

    @property
    def media_dir_path(self) -> Path:
        return super().media_dir_path / "clips"

    @property
    def load_input_data_files(self) -> str | Sequence[str] | Mapping[str, str | Sequence[str]] | None:
        return [str(self._archive_root / f"{self.split}.tsv")]

    def load_secondary_data(self) -> None:
        self._load_audio_durations(str(self._archive_root / "clip_durations.tsv"))

    def _load_audio_durations(self, durations_path: str) -> None:
        with open(durations_path, newline="", encoding="utf-8") as f:
            # noinspection PyTypeChecker
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            self._audio_durations = {row["clip"]: int(row["duration[ms]"]) / 1000 for row in reader}

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch = super().transform_batch(batch)
        batch.update({
            "audio_duration": [self._get_audio_duration(p, a) for p, a in zip(batch["path"], batch["audio"])]
        })
        return batch

    def _get_audio_duration(self, path: str, audio: Any) -> Any:
        duration = self._audio_durations.get(path, None)
        if duration is None:
            duration = AudioDecoder(audio).metadata.duration_seconds
        return duration
