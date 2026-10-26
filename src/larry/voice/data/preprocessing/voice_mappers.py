import io
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import av
import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict
from datasets.features.features import Features, FeatureType, List, Value
from torch.nn.utils.rnn import pad_sequence
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from larry.common.config.data.preprocessing.mapper_configs import MapperConfig
from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.data.preprocessing.mappers import Mapper, SingleColumnMapper
from larry.common.data.preprocessing.url_mappers import UrlMapper
from larry.common.data.utils import bot_user_agent, features_of
from larry.common.utils.types import EncodedAudio
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer, TextTokenizer
from larry.voice.config.preprocessing.voice_configs import AlignerConfig
from larry.voice.config.preprocessing.voice_mapper_configs import (
    AudioDurationMapperConfig,
    CTCScoreMapperConfig,
    LarrySpeechEncoderASRMapperConfig,
    LarrySpeechEncoderUnitIdExtractingMapperConfig,
    LarrySpeechEncoderUsingMapperConfig,
    SpeechSegmentationAligningMapperConfig,
    UrlAudioMapperConfig,
    WorldFeatureExtractingMapperConfig,
)
from larry.voice.data.preprocessing.alignment import Aligner, LongAlignment, TokenTimeline
from larry.voice.data.preprocessing.normalization import TextNormalizer
from larry.voice.data.preprocessing.segmentation import Segmenter
from larry.voice.data.preprocessing.transcript import NormalizedTokens, TokenStream, Transcript, Turn
from larry.voice.features.world import WorldAnalyzer
from larry.voice.model.speech_encoder import SpeechEncoderModel
from larry.voice.model.voice_models import SpeechEncoderModelOutput
from larry.voice.utils import bytes_to_waveforms, collapse_ctc_emissions

_ALIGNERS: dict[str, Aligner] = {}
_NORMALIZERS: dict[str, TextNormalizer] = {}
_ENCODERS: dict[str, SpeechEncoderModel] = {}
_TOKENIZERS: dict[str, TextTokenizer] = {}
_WORLD_ANALYZERS: dict[str, WorldAnalyzer] = {}


def _lazy_get_aligner(aligner_config: AlignerConfig, device: torch.device) -> Aligner:
    """For caching/obtaining ``Aligner`` instances; avoids class fields that would impact fingerprinting."""
    key = repr(aligner_config)
    if key not in _ALIGNERS:
        _ALIGNERS[key] = Aligner(aligner_config, device=device)
    return _ALIGNERS[key]


def _lazy_get_normalizer(config: MapperConfig, make_normalizer: Callable[[], TextNormalizer]) -> TextNormalizer:
    """For caching/obtaining ``TextNormalizer`` instances; avoids class fields that would impact fingerprinting."""
    key = repr(config)
    if key not in _NORMALIZERS:
        _NORMALIZERS[key] = make_normalizer()
    return _NORMALIZERS[key]


def _lazy_get_encoder(key: str, device: torch.device) -> SpeechEncoderModel:
    """For caching/obtaining ``SpeechEncoderModel`` instances; avoids class fields that would impact fingerprinting."""
    if key not in _ENCODERS:
        _ENCODERS[key] = SpeechEncoderModel.from_pretrained(key).to(device)
    return _ENCODERS[key]


def _lazy_get_tokenizer(key: str) -> TextTokenizer:
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = SentencePieceTokenizer(model_file=key)
    return _TOKENIZERS[key]


def _lazy_get_world_analyzer(config: WorldFeatureExtractingMapperConfig) -> WorldAnalyzer:
    key = repr(config)
    if key not in _WORLD_ANALYZERS:
        _WORLD_ANALYZERS[key] = WorldAnalyzer(config)

    return _WORLD_ANALYZERS[key]


class UrlAudioMapper(UrlMapper[EncodedAudio, UrlAudioMapperConfig]):
    """Implementation of ``UrlMapper`` for obtaining raw audio files from URLs."""

    def user_agent(self) -> str:
        """Sets the user agent to something that points to pipelines using this mapper in this project."""
        return bot_user_agent("larry-audio-downloader")

    def extract_content(self, example: str, content: bytes) -> EncodedAudio:
        """Wraps content in a typed dict for audio as bytes and a local file path."""
        return EncodedAudio(bytes=content, path=example)

    def validate_content(self, example: str, content: bytes) -> None:
        """Validates the audio is real by only materializing the first second of audio."""
        decoder = AudioDecoder(content)  # raises if the stream can't be parsed
        head = decoder.get_samples_played_in_range(0.0, 1.0)
        if head.data.shape[-1] == 0:
            raise ValueError(f"no decodable audio at start of {example}")

    def output_feature(self) -> FeatureType | None:
        """Sets the output feature to make sure only raw audio bytes are saved."""
        return Audio(decode=False)


class AudioDurationMapper(SingleColumnMapper[EncodedAudio | None, float | None, AudioDurationMapperConfig]):
    """Implementation of ``Mapper`` that gets the length of an audio clip and returns that as a new column."""

    def preprocess_example(self, example: EncodedAudio | None) -> float | None:
        """Given an audio example, decodes it and returns its length in seconds rounder to 3 decimal places."""
        if example is None:
            return None

        with av.open(io.BytesIO(example["bytes"]), mode="r") as c:
            s = c.streams.audio[0]
            n = sum(p.duration for p in c.demux(s) if p.duration)
            return round(float(n * s.time_base), 3)

    def output_feature(self) -> FeatureType | None:
        """Ensures precision is kept."""
        return Value("float64")


class SpeechSegmentationAligningMapper[
C: SpeechSegmentationAligningMapperConfig = SpeechSegmentationAligningMapperConfig
](Mapper[C], ABC):
    """An abstract mapper class for general transcript segmentation/alignment on large audio and text example pairs."""

    def __init__(self, provenance: Provenance, config: C) -> None:
        """Sets up the configured stage direction pattern and simple init."""
        super().__init__(provenance, config)
        self.stage_pattern = None
        if self.config.stage_direction_pattern:
            self.stage_pattern = re.compile(self.config.stage_direction_pattern)

    def validate(self) -> None:
        """Validates the relevant configs all work in the same sample rate."""
        super().validate()
        if self.config.sample_rate != self.config.aligner_config.sample_rate:
            raise ValueError("sample_rate must match aligner_config.sample_rate")

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """Overrides the preprocess_dataset method to perform very specific and necessary manipulation of column data.

        Specifically, all original columns need to be dropped because they contain the wrong row count for the expanded
        dataset. Then, each copy_column configured for each row is copied/duplicated onto every "child" row of generated
        segments/alignments. This retains provenance for all rows while sacrificing some storage for duplicated data.
        """
        return dataset.map(
            self._preprocess_batch,
            with_rank=True,
            features=self.output_features(dataset),
            **{**self.map_kwargs(), "remove_columns": list(features_of(dataset).keys())},
        )

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Reuses a lot of the transcript parsing code to first parse into segments.

        First turns the full transcript back into ``Turn`` objects using the shortcut parser, then uses each turn as the
        alignment target within the overall transcript/audio to find the segment to cut out and store as a new segment.
        """
        batch_results = {k: [] for k in self.config.copy_columns + self.output_columns}

        audios = batch[self.config.audio_column]
        texts = batch[self.config.text_column]

        aligner = _lazy_get_aligner(self.config.aligner_config, self.device)
        normalizer = _lazy_get_normalizer(self.config, self.make_normalizer)

        for i, (audio, text) in enumerate(zip(audios, texts, strict=True)):
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
            result = LongAlignment(self.config, aligner, emissions, stream.words, stream.word_turn_starts()).run()
            if stream.choose_readings(aligner, emissions.cpu(), result.spans, self.config.reading_margin_seconds):
                stream.rebuild()
                result = LongAlignment(
                    self.config, aligner, emissions, stream.words, stream.word_turn_starts()
                ).run(result)

            timeline = TokenTimeline(stream.fold(result.spans), stream.ftr.groups)
            timeline.fill_gaps()

            self.log.info(f"n_failed_words={result.n_failed_words} for {self.provenance_string(batch, i)}")
            self.log.info(f"greedy_decode_score={result.greedy_decode_score} for {self.provenance_string(batch, i)}")
            self.log.info(f"n_unspoken_words={sum(result.unspoken)} for {self.provenance_string(batch, i)}")

            segments = Segmenter(self.config, stream, timeline, stream.fold_flags(result.unspoken)).segments(waveform)

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
        """General transcript usage requires all of these calls so this helper method wraps new_transcript with them."""
        tr = self.new_transcript(turns, batch, index)
        tr.body_lines = tr.extract_body_lines(tr.raw_text)
        tr.load_appearances()
        return tr

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        """Manually sets features for all known output columns, including copied columns from parent rows."""
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
        """Adds the aligner model to the ancestry of the dataset being processed, since the aligner uses this model."""
        self.provenance.ancestry.models.add(self.config.aligner_config.model_id)

    @abstractmethod
    def make_normalizer(self) -> TextNormalizer:
        """Allows for overriding to build any implementation of ``TextNormalizer``."""
        ...

    @abstractmethod
    def new_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        """Allows for overriding to build any implementation of ``Transcript``."""
        ...


class CTCScoreMapper(Mapper[CTCScoreMapperConfig]):
    """Saves the ctc_mismatch score between a speech clip and its transcript as a new column."""

    def validate(self) -> None:
        """Validates the required input columns configuration."""
        super().validate()
        if not self.config.audio_column:
            raise ValueError(f"audio_column must be specified for {self.__class__.__name__}")
        if not self.config.text_column:
            raise ValueError(f"text_column must be specified for {self.__class__.__name__}")
        if not self.config.duration_column:
            raise ValueError(f"duration_column must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Uses an aligner model to score a batch of utterances against their ground-truth label transcripts."""
        aligner = _lazy_get_aligner(self.config.aligner_config, self.device)
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

    @staticmethod
    def make_normalizer() -> TextNormalizer:
        """Allows for overriding to make an instance of any ``TextNormalizer`` implementation."""
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
        """Obtains emissions and produces scores for a batch of utterances and their transcripts."""
        decoded: list[int] = []
        waveforms: list[torch.Tensor] = []
        for i in indices:
            try:
                waveforms.append(bytes_to_waveforms(audios[i]["bytes"], sample_rate=self.config.sample_rate)[0])
                decoded.append(i)
            except (RuntimeError, ValueError) as e:
                self.log.warning(f"skipping undecodable audio path={audios[i].get('path')!r}: {e}")

        indices = decoded
        if not indices:
            return

        emissions = aligner.clip_emissions(waveforms)
        for i, emission in zip(indices, emissions, strict=True):
            text = texts[i]
            if text is None:
                continue

            tokens = NormalizedTokens(text.split(), normalizer)
            if len(tokens.words) == 0:
                continue

            results = tokens.choose_readings_whole(aligner, emission)
            normalized[i] = " ".join(tokens.words)
            forced = results[-1] if results and results[-1] is not None else aligner.path_score(emission, tokens.words)

            if forced is not None:
                scores[i] = aligner.mismatch_score(emission, forced)

    def mutate_provenance(self) -> None:
        """Adds the aligner model to the provenance ancestry since this dataset will have been processed using it."""
        self.provenance.ancestry.models.add(self.config.aligner_config.model_id)

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        """Sets output features for the score column and optionally the normalized text column."""
        if self.config.text_output_column is not None:
            return Features({
                self.config.ctc_mismatch_output_column: Value("float64"),
                self.config.text_output_column: Value("string"),
            })
        return Features({self.config.ctc_mismatch_output_column: Value("float64")})


class LarrySpeechEncoderUsingMapper[C: LarrySpeechEncoderUsingMapperConfig = LarrySpeechEncoderUsingMapperConfig](
    Mapper[C], ABC
):
    def mutate_provenance(self) -> None:
        self.provenance.ancestry.models.add(self.config.speech_encoder_checkpoint_path)

    def _extract_waveforms(
            self,
            batch: dict[str, list[Any]],
            column: str,
            device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audios: list[EncodedAudio] = batch[column]
        waveforms = []
        waveform_lengths = []
        for audio in audios:
            waveform = bytes_to_waveforms(audio["bytes"], self.config.sample_rate)[0]
            waveforms.append(waveform)
            waveform_lengths.append(len(waveform))

        waveforms = pad_sequence(waveforms, batch_first=True).to(device)
        waveform_lengths = torch.tensor(waveform_lengths, dtype=torch.long, device=device)
        return waveforms, waveform_lengths


class LarrySpeechEncoderUnitIdExtractingMapper(
    LarrySpeechEncoderUsingMapper[LarrySpeechEncoderUnitIdExtractingMapperConfig]
):
    def __init__(self, provenance: Provenance, config: LarrySpeechEncoderUnitIdExtractingMapperConfig) -> None:
        super().__init__(provenance, config)

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model = _lazy_get_encoder(self.config.speech_encoder_checkpoint_path, self.device)

        device = next(model.parameters()).device
        waveforms, waveform_lengths = self._extract_waveforms(batch, self.config.audio_column, device)

        with torch.inference_mode():
            unit_ids, frame_lengths = model.tokenize(waveforms, waveform_lengths)

        return {
            self.config.unit_ids_output_column: [
                u[:n].astype(np.int16)
                for u, n in zip(unit_ids.cpu().numpy(), frame_lengths.tolist(), strict=True)
            ],
            self.config.frame_lengths_output_column: frame_lengths.cpu().numpy().tolist(),
        }

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return Features({
            self.config.unit_ids_output_column: List(Value("int16")),
            self.config.frame_lengths_output_column: Value("int32"),
        })


class LarrySpeechEncoderASRMapper(
    LarrySpeechEncoderUsingMapper[LarrySpeechEncoderASRMapperConfig]
):
    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model = _lazy_get_encoder(self.config.speech_encoder_checkpoint_path, self.device)
        tokenizer = _lazy_get_tokenizer(self.config.text_tokenizer_path)

        device = next(model.parameters()).device
        waveforms, waveform_lengths = self._extract_waveforms(batch, self.config.input_column, device)

        with torch.inference_mode():
            model_output: SpeechEncoderModelOutput = model(waveforms, waveform_lengths)

        greedy = model_output.logits.argmax(dim=-1).cpu()

        texts = []
        for i in range(greedy.shape[0]):
            texts.append(tokenizer.decode(collapse_ctc_emissions(
                greedy[i, :int(model_output.frame_lengths[i])],
                blank=model.config.vocab_size
            )))

        return {
            self.config.output_column: texts
        }

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return Features({
            self.config.output_column: Value("string"),
        })

    def mutate_provenance(self) -> None:
        super().mutate_provenance()
        self.provenance.ancestry.models.add(self.config.text_tokenizer_path)


class WorldFeatureExtractingMapper(Mapper[WorldFeatureExtractingMapperConfig]):
    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        analyzer = _lazy_get_world_analyzer(self.config)
        features = [
            analyzer.analyze(bytes_to_waveforms(audio["bytes"], self.config.sample_rate)[0])
            for audio in batch[self.config.audio_column]
        ]
        return {
            self.config.f0_output_column: [f.f0 for f in features],
            self.config.aperiodicity_output_column: [f.aperiodicity for f in features],
        }

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        return Features({
            self.config.f0_output_column: List(Value("float32")),
            self.config.aperiodicity_output_column: List(List(Value("float32"))),
        })
