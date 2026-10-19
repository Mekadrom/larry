from typing import Any

import torch

from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig
from larry.voice.data.preprocessing.alignment import SpanScore, TokenTimeline
from larry.voice.data.preprocessing.normalization import TextNormalizer
from larry.voice.data.preprocessing.segmentation import Segment, Segmenter
from larry.voice.data.preprocessing.transcript import FlattenedTranscript, TokenStream

GOOD = -0.1
BAD = -8.0


class Script:
    def __init__(self, word_seconds: float = 0.3, word_gap: float = 0.05, turn_gap: float = 0.3) -> None:
        self.word_seconds = word_seconds
        self.word_gap = word_gap
        self.turn_gap = turn_gap
        self.ftr = FlattenedTranscript()
        self.times: list[SpanScore | None] = []
        self.unspoken: list[bool] = []
        self.clock = 0.0

    def turn(self, speaker: str, text: str, score: float = GOOD, word_seconds: float | None = None,
             scores: dict[int, float] | None = None, unspoken: set[int] | None = None) -> Script:
        tokens = text.split()
        group = len(self.ftr.labels) + 1
        for k, token in enumerate(tokens):
            per_char = (scores or {}).get(k, score)
            seconds = word_seconds if word_seconds is not None else self.word_seconds
            if k in (unspoken or set()):
                self.times.append(None)
                self.unspoken.append(True)
            else:
                count = len(token)
                self.times.append(SpanScore(self.clock, self.clock + seconds, per_char * count, count))
                self.unspoken.append(False)
                self.clock += seconds + self.word_gap
            self.ftr.tokens.append(token)
            self.ftr.groups.append(group)
            self.ftr.turn_idx.append(len(self.ftr.labels))
            self.ftr.speakers.append(speaker)
            self.ftr.roles.append("speaker")
        self.ftr.labels.append(speaker)
        self.clock += self.turn_gap
        return self

    def segments(self, **config: Any) -> list[Segment]:
        stream = TokenStream(self.ftr, TextNormalizer())
        timeline = TokenTimeline(self.times, self.ftr.groups)
        timeline.fill_gaps()
        waveform = torch.zeros(int((self.clock + 1.0) * 16000))
        segmenter = Segmenter(SpeechSegmentationAligningMapperConfig(**config), stream, timeline, self.unspoken)
        return segmenter.segments(waveform)


def _texts(segments: list[Segment]) -> list[str]:
    return [segment.text for segment in segments]


def test_unspoken_tokens_are_left_out_and_split_their_turn() -> None:
    script = Script().turn("a", "one two three four five six", unspoken={2, 3})

    assert _texts(script.segments()) == ["one two", "five six"]


def test_a_fully_unspoken_turn_produces_no_segment() -> None:
    script = Script().turn("a", "wrong case called", unspoken={0, 1, 2}).turn("b", "the right case now")

    assert _texts(script.segments()) == ["the right case now"]


def test_poorly_aligned_edge_words_next_to_a_bad_segment_are_trimmed_without_padding_into_them() -> None:
    script = (
        Script()
        .turn("a", "you don't look like him", score=BAD)
        .turn("b", "accepting federal funds for its prisons", scores={0: -5.0})
    )

    segments = script.segments()

    assert _texts(segments) == ["you don't look like him", "federal funds for its prisons"]
    accepting = script.times[5]
    assert accepting is not None
    assert segments[1].start_s == accepting.end


def test_poorly_aligned_edge_words_away_from_bad_segments_are_kept() -> None:
    script = (
        Script()
        .turn("a", "you don't look like him")
        .turn("b", "accepting federal funds for its prisons", scores={0: -5.0})
    )

    assert _texts(script.segments()) == ["you don't look like him", "accepting federal funds for its prisons"]


def test_edges_next_to_dropped_unspoken_text_are_trimmed() -> None:
    script = Script().turn("a", "oh sorry federal funds for its prisons", unspoken={0, 1}, scores={2: -5.0})

    assert _texts(script.segments()) == ["funds for its prisons"]


def test_a_burst_of_short_crosstalk_turns_is_dropped_as_a_unit() -> None:
    script = Script(turn_gap=0.1).turn("a", "a long question about the statute and its history")
    for speaker, text in [("b", "well"), ("a", "okay no"), ("b", "but"), ("a", "right"), ("b", "I think")]:
        script.turn(speaker, text, word_seconds=0.25)
    script.turn("a", "and what about the second question here")

    assert _texts(script.segments()) == [
        "a long question about the statute and its history",
        "and what about the second question here",
    ]


def test_fewer_short_turns_than_the_minimum_are_kept() -> None:
    script = Script(turn_gap=0.1).turn("a", "a long question about the statute")
    for speaker, text in [("b", "well"), ("a", "okay")]:
        script.turn(speaker, text, word_seconds=0.25)

    assert _texts(script.segments()) == ["a long question about the statute", "well", "okay"]


def test_short_turns_spread_over_a_long_stretch_are_not_crosstalk() -> None:
    script = Script(turn_gap=4.0)
    for speaker, text in [("a", "yes"), ("b", "no"), ("a", "maybe"), ("b", "fine")]:
        script.turn(speaker, text)

    assert _texts(script.segments()) == ["yes", "no", "maybe", "fine"]
