import dataclasses
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.transforms
from datasets import Dataset, Audio, Array2D, Features
from torchaudio import transforms
from torchcodec.decoders import AudioDecoder

from larry.common.config.preprocessing_mapper_configs import PreprocessingMapperConfig
from larry.common.data.preprocessing.preprocessing_mappers import PreprocessingMapper, PruningPreprocessingMapper
from larry.voice.config.voice_preprocessing_mapper_configs import MelExtractingPreprocessingMapperConfig, \
    AudioColumnCastPreprocessingMapperConfig, AudioDurationColumnPruningPreprocessingMapperConfig, \
    VoicePreprocessingMapperConfig, AudioQualityPruningPreprocessingMapperConfig


class VoicePreprocessingMapper[I = None, O = None](PreprocessingMapper[I, O], ABC):
    config: VoicePreprocessingMapperConfig

    def __init__(self, config: VoicePreprocessingMapperConfig) -> None:
        super().__init__(config=config)


class AudioDurationColumnPruningPreprocessingMapper(PruningPreprocessingMapper[int | float]):
    config: AudioDurationColumnPruningPreprocessingMapperConfig

    def __init__(self, config: AudioDurationColumnPruningPreprocessingMapperConfig) -> None:
        super().__init__(config=config)
        self.config = config

    def prune_example(self, example: int | float) -> bool:
        return example < self.config.min_duration or example > self.config.max_duration


class AudioColumnCastPreprocessingMapper(VoicePreprocessingMapper[dict[str, Any], torch.Tensor]):
    config: AudioColumnCastPreprocessingMapperConfig

    def __init__(self, config: AudioColumnCastPreprocessingMapperConfig) -> None:
        super().__init__(config)

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        dataset = dataset.cast_column(
            self.config.input_column,
            Audio(sampling_rate=self.config.sample_rate, decode=self.config.decode)
        )
        return super().preprocess_dataset(dataset)

    def preprocess_example(self, example: dict[str, Any]) -> torch.Tensor:
        decoder = AudioDecoder(
            example.get("bytes") or Path(example.get("path")).read_bytes(),
            sample_rate=self.config.sample_rate
        )
        return decoder.get_all_samples().data


class AudioQualityPruningPreprocessingMapper(PruningPreprocessingMapper[torch.Tensor]):
    config: AudioQualityPruningPreprocessingMapperConfig

    def __init__(self, config: AudioQualityPruningPreprocessingMapperConfig):
        super().__init__(config=config)
        self.config = config

    def prune_example(self, example: torch.Tensor) -> bool:
        waveform = example.mean(dim=0)

        frame, hop = int(0.025 * self.config.sample_rate), int(0.010 * self.config.sample_rate)
        db = 10 * torch.log10(waveform.unfold(0, frame, hop).pow(2).mean(-1).clamp(min=1e-10))
        active = db > db.max() - 35
        noise_floor = torch.quantile(db, 0.10)
        speech_level = torch.quantile(db[active], 0.50)

        rms_dbfs = (10 * torch.log10(waveform.pow(2).mean().clamp(min=1e-10))).item()
        active_ratio = active.float().mean().item()
        snr_db = (speech_level - noise_floor).item()
        clip_frac = (waveform.abs() >= 0.999).float().mean().item()

        prune_rms_dbfs = rms_dbfs < self.config.min_rms_dbfs
        prune_active_ratio = active_ratio < self.config.min_rms_dbfs
        prune_snr = snr_db < self.config.min_snr
        prune_clip_frac = clip_frac > self.config.max_clipping_portion

        return any([prune_rms_dbfs, prune_active_ratio, prune_snr, prune_clip_frac])


class MelExtractingPreprocessingMapper(VoicePreprocessingMapper[torch.Tensor, np.ndarray]):
    config: MelExtractingPreprocessingMapperConfig

    def __init__(self, config: MelExtractingPreprocessingMapperConfig) -> None:
        super().__init__(config)
        kwargs = dataclasses.asdict(self.config)
        for field in dataclasses.fields(PreprocessingMapperConfig):
            del kwargs[field.name]
        del kwargs["audio_dtype"]
        self.transform = transforms.MelSpectrogram(norm="slaney", mel_scale="slaney", **kwargs)

    def preprocessing_features(self, dataset: Dataset) -> Features | None:
        features = dataset.features.copy()
        for col in self.remove_columns:
            features.pop(col, None)
        features.update({"mels": Array2D(shape=(None, self.config.n_mels), dtype=self.config.audio_dtype)})
        return features

    def preprocess_example(self, example: torch.Tensor) -> np.ndarray:
        if example.ndim > 1 and example.shape[0] == 2:
            # downmix to mono
            example = example.data.mean(dim=0, keepdim=True)

        if example.ndim > 1 and example.shape[0] == 1:
            # remove single channel
            example = example.squeeze(0)

        waveforms = self._remove_mains_hum(example)

        log_mel = torch.log(self.transform(waveforms).clamp(min=1e-5))

        return log_mel.T.contiguous().to(torch.float16).numpy()

    def _remove_mains_hum(self, waveform, frequencies=None):
        """Remove mains hum and harmonics."""
        if frequencies is None:
            frequencies = [60, 120, 180, 240]

        for freq in frequencies:
            waveform = torchaudio.functional.bandreject_biquad(
                waveform, self.config.sample_rate, central_freq=freq, Q=30.0
            )
        return waveform
