import csv
from pathlib import Path
from typing import Mapping, Any

from datasets import Features, Value, Dataset
from torchcodec.decoders import AudioDecoder

from larry.common.data.cached_dataset_providers import MediaCachedDatasetProvider, CachedDatasetProvider
from larry.common.data.dataset_providers import DatasetProvider
from larry.voice.config.preprocessing.voice_dataset_configs import CommonVoiceDatasetConfig, \
    SCOTUSManifestCachedDatasetConfig


class CommonVoiceMediaCachedDatasetProvider(MediaCachedDatasetProvider[CommonVoiceDatasetConfig]):
    _archive_root: Path
    _audio_durations: dict[str, float]

    def __init__(self, config: CommonVoiceDatasetConfig) -> None:
        super().__init__(config)
        self._archive_root = self.download_extract_cache_dir / self.config.archive_root

    @property
    def media_dir_path(self) -> Path:
        return super().media_dir_path / "clips"

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return {"train": str(self._archive_root / f"{self.split}.tsv")}

    def make_features(self, data_files: Mapping[str, str]) -> Features | None:
        file = data_files["train"]
        with open(file, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        self.log.info(f"data_file={file} header={header}")
        return Features({name: Value("string") for name in header})

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


class SCOTUSManifestCachedDatasetProvider(CachedDatasetProvider[SCOTUSManifestCachedDatasetConfig]):
    _archive_root: Path

    def __init__(self, config: SCOTUSManifestCachedDatasetConfig) -> None:
        super().__init__(config)
        self._archive_root = self.download_extract_cache_dir

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        return {"train": str(self._archive_root / self.config.docket_manifest_file_name)}

    def cast_cols(self, dataset: Dataset) -> Dataset:
        return dataset

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return batch
