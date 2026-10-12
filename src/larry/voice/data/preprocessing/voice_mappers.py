import io
import re
from abc import abstractmethod, ABC
from collections.abc import Callable
from typing import Any

import av
from datasets import Dataset, Audio, DatasetDict
from datasets.features.features import FeatureType, Value, Features
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import UrlMapper, Mapper, SingleColumnMapper
from larry.common.data.utils import features_of
from larry.common.utils.types import EncodedAudio
from larry.voice.config.preprocessing.voice_configs import AlignerConfig
from larry.voice.config.preprocessing.voice_mapper_configs import UrlAudioMapperConfig, AudioDurationMapperConfig, \
    SpeechSegmentationAligningMapperConfig, \
    CTCScoreMapperConfig
from larry.voice.data.preprocessing.alignment import LongAlignment, TokenTimeline, Aligner
from larry.voice.data.preprocessing.normalization import TextNormalizer
from larry.voice.data.preprocessing.segmentation import Segmenter
from larry.voice.data.preprocessing.transcript import TokenStream, Turn, Transcript, NormalizedTokens
from larry.voice.utils import bytes_to_waveforms

_ALIGNERS: dict[str, Aligner] = {}
_NORMALIZERS: dict[str, TextNormalizer] = {}


def _lazy_get_aligner(aligner_config: AlignerConfig, device_index: int) -> Aligner:
    key = repr(aligner_config)
    if key not in _ALIGNERS:
        _ALIGNERS[key] = Aligner(aligner_config, device=f"cuda:{device_index}")
    return _ALIGNERS[key]


def _lazy_get_normalizer(config: MapperConfig, make_normalizer: Callable[[], TextNormalizer]) -> TextNormalizer:
    key = repr(config)
    if key not in _NORMALIZERS:
        _NORMALIZERS[key] = make_normalizer()
    return _NORMALIZERS[key]


class UrlAudioMapper(UrlMapper[EncodedAudio, UrlAudioMapperConfig]):
    def user_agent(self) -> str:
        return "larry-audio-downloader/1.0 (dataset research; contact via github.com/Mekadrom)"

    def extract_content(self, example: str, content: bytes) -> EncodedAudio:
        return EncodedAudio(bytes=content, path=example)

    def validate_content(self, example: str, content: bytes) -> None:
        decoder = AudioDecoder(content)  # raises if the stream can't be parsed
        head = decoder.get_samples_played_in_range(0.0, 1.0)
        if head.data.shape[-1] == 0:
            raise ValueError(f"no decodable audio at start of {example}")

    def output_feature(self) -> FeatureType | None:
        return Audio(decode=False)


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
    def __init__(self, provenance: Provenance, config: C) -> None:
        super().__init__(provenance, config)
        self.stage_pattern = None
        if self.config.stage_direction_pattern:
            self.stage_pattern = re.compile(self.config.stage_direction_pattern)

    def validate(self) -> None:
        super().validate()
        if self.config.sample_rate != self.config.aligner_config.sample_rate:
            raise ValueError("sample_rate must match aligner_config.sample_rate")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        return dataset.map(
            self._preprocess_batch,
            with_rank=True,
            features=self.output_features(dataset),
            **{**self.map_kwargs(), "remove_columns": list(features_of(dataset).keys())},
        )

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch_results = {k: [] for k in self.config.copy_columns + self.output_columns}

        audios = batch[self.config.audio_column]
        texts = batch[self.config.text_column]

        aligner = _lazy_get_aligner(self.config.aligner_config, self.device_index)
        normalizer = _lazy_get_normalizer(self.config, self.make_normalizer)

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
            if stream.choose_readings(aligner, emissions.cpu(), result.spans, self.config.reading_margin_seconds):
                stream.rebuild()
                result = LongAlignment(self.config, aligner, emissions, stream.words).run(result)
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
                batch_results[self.config.audio_output_column].append({"bytes": flac.numpy().tobytes(), "path": None})
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

    def output_features(self, dataset):
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

    def mutate_provenance(self) -> None:
        self.provenance.ancestry.models.add(self.config.aligner_config.model_id)

    @abstractmethod
    def make_normalizer(self) -> TextNormalizer:
        ...

    @abstractmethod
    def new_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        ...


class CTCScoreMapper(Mapper[CTCScoreMapperConfig]):
    def validate(self) -> None:
        super().validate()
        if not self.config.audio_column:
            raise ValueError(f"audio_column must be specified for {self.__class__.__name__}")
        if not self.config.text_column:
            raise ValueError(f"text_column must be specified for {self.__class__.__name__}")
        if not self.config.duration_column:
            raise ValueError(f"duration_column must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        aligner = _lazy_get_aligner(self.config.aligner_config, self.device_index)
        normalizer = _lazy_get_normalizer(self.config, self.make_normalizer)

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

        result: dict[str, Any] = {
            self.config.ctc_mismatch_output_column: scores,
        }
        text_output_column = self.config.text_output_column
        if text_output_column is not None:
            result[text_output_column] = normalized

        return result

    def make_normalizer(self) -> TextNormalizer:
        return TextNormalizer()

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

            results = tokens.choose_readings_whole(aligner, emission)
            normalized[i] = " ".join(tokens.words)
            if results and results[-1] is not None:
                forced = results[-1]
            else:
                forced = aligner.path_score(emission, tokens.words)
            if forced is not None:
                scores[i] = aligner.mismatch_score(emission, forced)

    def mutate_provenance(self) -> None:
        self.provenance.ancestry.models.add(self.config.aligner_config.model_id)

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        if self.config.text_output_column is not None:
            return Features({
                self.config.ctc_mismatch_output_column: Value("float64"),
                self.config.text_output_column: Value("string"),
            })
        return Features({self.config.ctc_mismatch_output_column: Value("float64")})
