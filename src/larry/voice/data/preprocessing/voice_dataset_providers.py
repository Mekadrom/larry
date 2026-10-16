import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value
from torchcodec.decoders import AudioDecoder

from larry.common.data.cached_dataset_providers import CachedDatasetProvider, MediaCachedDatasetProvider
from larry.voice.config.preprocessing.voice_dataset_configs import (
    CommonVoiceDatasetConfig,
    SCOTUSManifestCachedDatasetConfig,
    VCTKCachedDatasetProviderConfig,
)


class CommonVoiceMediaCachedDatasetProvider(MediaCachedDatasetProvider[CommonVoiceDatasetConfig]):
    """For extracting an already-downloaded Common Voice 27.0 archive and turning it into an HF dataset of parquets."""

    _archive_root: Path
    _audio_durations: dict[str, float]

    def __init__(self, config: CommonVoiceDatasetConfig) -> None:
        """Simple init."""
        super().__init__(config)
        self._archive_root = self.download_extract_cache_dir / self.config.archive_root

    @property
    def media_dir_path(self) -> Path:
        """Where audio clips is found after extraction."""
        return super().media_dir_path / "clips"

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        """Can use load_dataset to load TSV files and start the transformation into a materialized parquet dataset."""
        return {
            "train": str(self._archive_root / "train.tsv"),
            "dev": str(self._archive_root / "dev.tsv"),
            "test": str(self._archive_root / "test.tsv")
        }

    def make_features(self, data_files: Mapping[str, str]) -> Features | None:
        """Reads the header from the input TSV file and produces a Features dict describing each column as strings."""
        file = data_files["train"]
        with open(file, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

        self.log.info(f"data_file={file} header={header}")
        return Features({name: Value("string") for name in header})

    def load_secondary_data(self) -> None:
        """To avoid the audio durations needing to be read from metadata or materialize audio, load this TSV."""
        self._load_audio_durations(str(self._archive_root / "clip_durations.tsv"))

    def _load_audio_durations(self, durations_path: str) -> None:
        """Load and cache audio durations, for later joining on the materialized dataset."""
        with open(durations_path, newline="", encoding="utf-8") as f:
            # noinspection PyTypeChecker
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            self._audio_durations = {row["clip"]: int(row["duration[ms]"]) / 1000 for row in reader}

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Supermethod pulls the media by path into the batch, this implementation joins duration_s."""
        batch = super().transform_batch(batch)
        batch.update({
            "duration_s": [self._get_audio_duration(p, a) for p, a in zip(batch["path"], batch["audio"], strict=False)]
        })
        return batch

    def _get_audio_duration(self, path: str, audio: Any) -> Any:
        """If the duration is None for an example, materialize the audio and get it that way."""
        duration = self._audio_durations.get(path, None)
        if duration is None:
            duration = AudioDecoder(audio).metadata.duration_seconds

        return duration


class SCOTUSManifestCachedDatasetProvider(CachedDatasetProvider[SCOTUSManifestCachedDatasetConfig]):
    """Downloads and loads the SCOTUS manifest as a dataset with only URLs, no materialized audio."""

    _archive_root: Path

    def __init__(self, config: SCOTUSManifestCachedDatasetConfig) -> None:
        """Simple init."""
        super().__init__(config)
        self._archive_root = self.download_extract_cache_dir

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        """The single input file is a docket manifest file created by the ``SCOTUSDocketManifestDownloader``."""
        return {"train": str(self._archive_root / self.config.docket_manifest_file_name)}

    def cast_cols(self, dataset: Dataset) -> Dataset:
        """No-op."""
        return dataset

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """No-op."""
        return batch


class VCTKCachedDatasetProvider(MediaCachedDatasetProvider[VCTKCachedDatasetProviderConfig]):
    """Loads the VCTK dataset directly from the datashare archive."""

    @property
    def load_input_data_files(self) -> Mapping[str, str]:
        """The manifest for the datashare after extraction."""
        return {"train": str(self.download_extract_cache_dir / self.config.manifest_file_name)}

    def load_secondary_data(self) -> None:
        """The manifest is generated manually after extraction and prior to dataset creation."""
        self._write_manifest()

    def _write_manifest(self) -> None:
        """Manually creates a manifest in csv/tsv format for the dataset prior to parsing as an HF dataset."""
        manifest = self.download_extract_cache_dir / self.config.manifest_file_name
        if manifest.exists():
            return

        rows = []
        for txt in sorted((self.download_extract_cache_dir / "txt").glob("*/*.txt")):
            for suffix in ("mic1", "mic2"):
                speaker = txt.parent.name
                audio = (self.media_dir_path / speaker / f"{txt.stem}_{suffix}.flac").resolve()
                if not audio.exists():
                    continue

                text = " ".join(txt.read_text(encoding="utf-8").split())
                rows.append((txt.stem, speaker, str(audio.relative_to(self.media_dir_path)), text, suffix))

        tmp = manifest.with_name(manifest.name + ".part")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, lineterminator="\n")
            writer.writerow(["id", "speaker_id", "path", "text_original", "mic_id"])
            writer.writerows(rows)

        tmp.replace(manifest)

    def transform_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Supermethod materializes audio, this implementation adds duration by reading the materialized audio."""
        batch = super().transform_batch(batch)
        batch["duration_s"] = [AudioDecoder(path).metadata.duration_seconds for path in batch["audio"]]
        return batch
