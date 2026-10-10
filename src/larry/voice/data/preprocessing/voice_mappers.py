import io
import re
from abc import abstractmethod, ABC
from typing import Any

import av
import numpy as np
import torch
import torchaudio
from datasets import Array2D, Dataset, DatasetDict, Audio
from datasets.features.features import FeatureType, Value, Features
from torchaudio import transforms
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from larry.common.data.preprocessing.mappers import SingleColumnMapper, UrlMapper, Mapper
from larry.common.data.utils import features_of
from larry.common.utils.types import EncodedAudio
from larry.voice.config.preprocessing.voice_mapper_configs import MelExtractingMapperConfig, \
    VoiceMapperConfig, UrlAudioMapperConfig, AudioDurationMapperConfig, SpeechSegmentationAligningMapperConfig, \
    CTCScoreMapperConfig
from larry.voice.data.preprocessing.alignment import LongAlignment, TokenTimeline, Aligner
from larry.voice.data.preprocessing.normalization import TextNormalizer
from larry.voice.data.preprocessing.segmentation import Segmenter
from larry.voice.data.preprocessing.transcript import TokenStream, Turn, Transcript, NormalizedTokens
from larry.voice.utils import bytes_to_waveforms

_ALIGNERS: dict[str, Aligner] = {}
_NORMALIZERS: dict[str, TextNormalizer] = {}


class VoiceMapper[I, O, C: VoiceMapperConfig = VoiceMapperConfig](SingleColumnMapper[I, O, C]):
    def output_feature(self) -> FeatureType | None:
        return Audio(decode=False)


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
        head = decoder.get_samples_played_in_range(0.0, 1.0)
        if head.data.shape[-1] == 0:
            raise ValueError(f"no decodable audio at start of {example}")


class AudioDurationMapper(SingleColumnMapper[EncodedAudio | None, float | None, AudioDurationMapperConfig]):
    def preprocess_example(self, example: EncodedAudio | None) -> float | None:
        if example is None:
            return None
        with av.open(io.BytesIO(example["bytes"]), mode="r") as c:
            s = c.streams.audio[0]
            n = sum(p.duration for p in c.demux(s) if p.duration)
            return round(float(n * s.time_base), 3)

    def output_feature(self) -> FeatureType | None:
        return Value("float64")


class SpeechSegmentationAligningMapper[
C: SpeechSegmentationAligningMapperConfig = SpeechSegmentationAligningMapperConfig
](Mapper[C], ABC):
    def __init__(self, provenance_columns: list[str], config: C) -> None:
        super().__init__(provenance_columns, config)
        self.output_columns = [
            "audio",
            "text_original",
            "text_normalized",
            "duration_s",
            "start_s",
            "end_s",
            "speaker_id",
            "avg_word_score"
        ]

        self.stage_pattern = None
        if self.config.stage_direction_pattern:
            self.stage_pattern = re.compile(self.config.stage_direction_pattern)

    def validate(self) -> None:
        if self.config.audio_column is None:
            raise ValueError(f"audio_column must be specified for {self.__class__.__name__}")
        if self.config.text_column is None:
            raise ValueError(f"text_column must be specified for {self.__class__.__name__}")
        if self.config.sample_rate != self.config.aligner_config.sample_rate:
            raise ValueError("sample_rate must match aligner_config.sample_rate")

    def preprocess_dataset(self, dataset):
        self.remove_columns = list(features_of(dataset).keys())
        return dataset.map(
            self.preprocess_batch,
            features=self.preprocessing_features(dataset),
            **self.map_kwargs()
        )

    def _lazy_get_aligner(self) -> Aligner:
        key = repr(self.config.aligner_config)
        if key not in _ALIGNERS:
            _ALIGNERS[key] = Aligner(self.config.aligner_config)
        return _ALIGNERS[key]

    def _lazy_get_normalizer(self) -> TextNormalizer:
        key = repr(self.config)
        if key not in _NORMALIZERS:
            _NORMALIZERS[key] = self.make_normalizer()
        return _NORMALIZERS[key]

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch_results = {k: [] for k in self.config.copy_columns + self.output_columns}

        audios = batch[self.config.audio_column]
        texts = batch[self.config.text_column]

        aligner = self._lazy_get_aligner()
        normalizer = self._lazy_get_normalizer()

        for i, (audio, text) in enumerate(zip(audios, texts)):
            waveform = bytes_to_waveforms(audio["bytes"], self.config.sample_rate)[0]
            emissions = aligner.ctc_emissions(waveform)

            turns = []
            for line_no, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                speaker, _, said = line.partition(": ")
                if self.stage_pattern is not None:
                    said = self.stage_pattern.sub(" ", said)
                turns.append(Turn(speaker, said, line_no))

            stream = TokenStream.from_transcript(self.make_transcript(turns, batch, i), normalizer)
            result = LongAlignment(self.config, aligner, emissions, stream.words).run()
            if stream.choose_readings(aligner, emissions, result.spans, self.config.reading_margin_seconds):
                stream.rebuild()
                result = LongAlignment(self.config, aligner, emissions, stream.words).run()
            timeline = TokenTimeline(stream.fold(result.spans), stream.ftr.groups)
            timeline.fill_gaps()

            self.log.info(f"n_failed_words={result.n_failed_words} for {self.provenance_string(batch, i)}")
            self.log.info(f"greedy_decode_score={result.greedy_decode_score} for {self.provenance_string(batch, i)}")

            segments = Segmenter(self.config, stream, timeline).segments(waveform)

            if not segments:
                self.log.warning(f"skipping {self.provenance_string(batch, i)}: no segments survived")
                continue

            for seg in segments:
                flac = AudioEncoder(
                    seg.audio.unsqueeze(0).cpu().float(),
                    sample_rate=self.config.sample_rate
                ).to_tensor(format="flac")
                batch_results["audio"].append({"bytes": flac.numpy().tobytes(), "path": None})
                batch_results["text_original"].append(seg.text)
                batch_results["text_normalized"].append(seg.text_normalized)
                batch_results["duration_s"].append(round(seg.end_s - seg.start_s, 3))
                batch_results["start_s"].append(round(seg.start_s, 3))
                batch_results["end_s"].append(round(seg.end_s, 3))
                batch_results["speaker_id"].append(seg.speaker_id)
                batch_results["avg_word_score"].append(seg.avg_word_score)

            for k in self.config.copy_columns:
                batch_results[k].extend([batch[k][i]] * len(segments))

        return batch_results

    def make_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        tr = self.new_transcript(turns, batch, index)
        tr.body_lines = tr.extract_body_lines(tr.raw_text)
        tr.load_appearances()
        return tr

    @abstractmethod
    def make_normalizer(self) -> TextNormalizer:
        ...

    @abstractmethod
    def new_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        ...

    def preprocessing_features(self, dataset):
        src = features_of(dataset).copy()
        return Features({
            **{c: src[c] for c in self.config.copy_columns},
            "audio": Audio(decode=False),
            "text_original": Value("string"),
            "text_normalized": Value("string"),
            "duration_s": Value("float64"),
            "start_s": Value("float64"),
            "end_s": Value("float64"),
            "speaker_id": Value("string"),
            "avg_word_score": Value("float64"),
        })


class CTCScoreMapper(Mapper[CTCScoreMapperConfig]):
    def validate(self) -> None:
        if not self.config.audio_column:
            raise ValueError(f"audio_column must be specified for {self.__class__.__name__}")
        if not self.config.text_column:
            raise ValueError(f"text_column must be specified for {self.__class__.__name__}")
        if not self.config.duration_column:
            raise ValueError(f"duration_column must be specified for {self.__class__.__name__}")

    def _lazy_get_aligner(self) -> Aligner:
        key = repr(self.config.aligner_config)
        if key not in _ALIGNERS:
            _ALIGNERS[key] = Aligner(self.config.aligner_config)
        return _ALIGNERS[key]

    def _lazy_get_normalizer(self) -> TextNormalizer:
        key = repr(self.config)
        if key not in _NORMALIZERS:
            _NORMALIZERS[key] = TextNormalizer()
        return _NORMALIZERS[key]

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        aligner = self._lazy_get_aligner()
        normalizer = self._lazy_get_normalizer()

        audios = batch[self.config.audio_column]
        texts = batch[self.config.text_column]
        durations = batch[self.config.duration_column]

        order = sorted(range(len(audios)), key=lambda i: durations[i])
        scores: list[float | None] = [None] * len(audios)
        normalized: list[str | None] = [None] * len(audios)

        group: list[int] = []
        for i in order:
            # ascending order: durations[i] is the longest in the group, so this is the padded size
            padded_seconds = (len(group) + 1) * durations[i]
            if len(group) > 0 and padded_seconds > self.config.max_batch_seconds:
                self._score_group(aligner, normalizer, group, audios, texts, scores, normalized)
                group = []
            group.append(i)

        if len(group) > 0:
            self._score_group(aligner, normalizer, group, audios, texts, scores, normalized)

        return {
            self.config.score_output_column: scores,
            self.config.text_output_column: normalized,
        }

    def _score_group(
            self,
            aligner: Aligner,
            normalizer: TextNormalizer,
            indices: list[int],
            audios: list[Any],
            texts: list[str],
            scores: list[float | None],
            normalized: list[str | None],
    ) -> None:
        waveforms = [
            bytes_to_waveforms(audios[i]["bytes"], sample_rate=self.config.sample_rate)[0]
            for i in indices
        ]
        emissions = aligner.clip_emissions(waveforms)
        for i, emission in zip(indices, emissions):
            text = texts[i]
            if text is None:
                continue

            tokens = NormalizedTokens(text.split(), normalizer)
            if len(tokens.words) == 0:
                continue

            tokens.choose_readings_whole(aligner, emission)
            normalized[i] = " ".join(tokens.words)
            targets, _ = aligner.get_targets_owners(tokens.words)
            scores[i] = aligner.mismatch_score(emission, targets)

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        features = super().preprocessing_features(dataset)
        if not features:
            features = features_of(dataset).copy()
        for column in self.remove_columns:
            features.pop(column, None)
        features[self.config.output_column] = Value("float64")
        features["text_normalized"] = Value("string")
        return features


class MelExtractingMapper(VoiceMapper[EncodedAudio, np.ndarray | None, MelExtractingMapperConfig]):
    def __init__(self, provenance_columns: list[str], config: MelExtractingMapperConfig) -> None:
        super().__init__(provenance_columns, config)
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
