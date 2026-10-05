import datasets
import numpy as np
import requests
import torch
import torchaudio.transforms
from datasets import Array2D
from datasets.features.features import FeatureType
from torchaudio import transforms
from torchcodec.decoders import AudioDecoder

from larry.common.data.preprocessing.mappers import SingleColumnMapper, UrlMapper
from larry.common.utils.types import EncodedAudio
from larry.voice.config.preprocessing.voice_mapper_configs import MelExtractingMapperConfig, \
    VoiceMapperConfig, UrlAudioMapperConfig
from larry.voice.utils import bytes_to_waveforms


class VoiceMapper[I, O, C: VoiceMapperConfig = VoiceMapperConfig](SingleColumnMapper[I, O, C]):
    def output_feature(self) -> FeatureType | None:
        return datasets.Audio(decode=False)


class UrlAudioMapper(
    UrlMapper[EncodedAudio, UrlAudioMapperConfig],
    VoiceMapper[str, EncodedAudio | None, UrlAudioMapperConfig]
):
    def user_agent(self) -> str:
        return "larry-audio-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def extract_content(self, example: str, content: bytes) -> EncodedAudio:
        return EncodedAudio(bytes=content, path=example)

    def validate_content(self, example: str, content: bytes) -> None:
        decoder = AudioDecoder(content)  # raises if the stream can't be parsed
        meta = decoder.metadata
        duration = meta.duration_seconds_from_header
        if duration is None or duration < 0.1:
            raise ValueError(f"implausible duration: {duration} for url={example}")

        tail = decoder.get_samples_played_in_range(max(0.0, duration - 5.0), duration)
        if tail.data.shape[-1] == 0:
            raise ValueError(f"no decodable audio at the end of the file from {example}; likely truncated")


class MelExtractingMapper(VoiceMapper[EncodedAudio, np.ndarray | None, MelExtractingMapperConfig]):
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
