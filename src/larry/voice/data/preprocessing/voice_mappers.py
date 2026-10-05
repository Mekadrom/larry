import numpy as np
import torch
import torchaudio.transforms
from datasets import Array2D
from datasets.features.features import FeatureType
from torchaudio import transforms

from larry.common.data.preprocessing.mappers import SingleColumnMapper
from larry.common.utils.types import EncodedAudio
from larry.voice.config.preprocessing.voice_mapper_configs import MelExtractingMapperConfig, VoiceSingleColumnMapperConfig
from larry.voice.utils import bytes_to_waveforms


class VoiceSingleColumnMapper[I, O, C: VoiceSingleColumnMapperConfig = VoiceSingleColumnMapperConfig](SingleColumnMapper[I, O, C]):
    ...


class MelExtractingMapper(VoiceSingleColumnMapper[EncodedAudio, np.ndarray | None, MelExtractingMapperConfig]):
    def __init__(self, config: MelExtractingMapperConfig) -> None:
        super().__init__(config)
        self.transform = transforms.MelSpectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            norm="slaney",
            mel_scale="slaney"
        )

    def output_feature(self) -> FeatureType | None:
        return Array2D(shape=(None, self.config.n_mels), dtype=self.config.audio_dtype)

    def preprocess_example(self, example: EncodedAudio) -> np.ndarray | None:
        waveform = bytes_to_waveforms(example["bytes"], self.config.sample_rate)

        waveform = self._remove_mains_hum(waveform)

        log_mel = torch.log(self.transform(waveform).clamp(min=1e-5))

        return log_mel.T.contiguous().to(getattr(torch, self.config.audio_dtype)).numpy()

    def _remove_mains_hum(self, waveform, frequencies=None):
        """Remove mains hum and harmonics."""
        if frequencies is None:
            frequencies = [60, 120, 180, 240]

        for freq in frequencies:
            waveform = torchaudio.functional.bandreject_biquad(
                waveform, self.config.sample_rate, central_freq=freq, Q=30.0
            )
        return waveform
