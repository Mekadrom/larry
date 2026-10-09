import dataclasses
import logging

import torch

from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig
from larry.voice.data.preprocessing.alignment import TokenTimeline, SpanScore
from larry.voice.data.preprocessing.transcript import TokenStream


@dataclasses.dataclass
class Segment:
    audio: torch.Tensor
    text: str
    speaker_id: str
    avg_word_score: float
    start_s: float
    end_s: float


class Segmenter:
    def __init__(
            self,
            config: SpeechSegmentationAligningMapperConfig,
            stream: TokenStream,
            timeline: TokenTimeline,
    ) -> None:
        self.log = logging.getLogger(type(self).__name__)
        self.config = config
        self.stream = stream
        self.timeline = timeline

    def segments(self, waveform: torch.Tensor) -> list[Segment]:
        segments = []
        for toks in self.stream.turns():
            for piece in self._split_turn(toks, self.timeline.spans, self.stream.ftr.tokens):
                segment = self._make_segment(piece, waveform)
                if segment is not None:
                    segments.append(segment)
        return segments

    def _split_turn(self, toks: list[int], spans: list[SpanScore | None], tokens: list[str]) -> list[list[int]]:
        """Split a turn into pieces under max_segment_seconds, preferring sentence ends."""
        pieces = []
        cur = []
        prev = None
        for t in toks:
            cur_span = spans[t]
            if cur_span is None:
                continue

            gap = 0.0
            if prev is not None:
                prev_span = spans[prev]
                if prev_span:
                    gap = cur_span.start - prev_span.end

            pause = self.config.max_pause_seconds is not None and gap > self.config.max_pause_seconds
            too_long = bool(cur) and cur_span.end - spans[cur[0]].start > self.config.max_segment_seconds

            if cur and (pause or too_long):
                stops = [
                    k
                    for k, c in enumerate(cur)
                    if tokens[c].endswith((".", "?", "!"))
                ]

                if pause or not stops:
                    cut = len(cur)
                else:
                    cut = stops[-1] + 1

                # sentence cut leaves a remainder that's still too long with t; cut everything instead
                if cut < len(cur) and cur_span.end - spans[cur[cut]].start > self.config.max_segment_seconds:
                    cut = len(cur)

                pieces.append(cur[:cut])
                cur = cur[cut:]

            cur.append(t)
            prev = t

        if cur:
            pieces.append(cur)

        return pieces

    def _make_segment(self, piece: list[int], waveform: torch.Tensor) -> Segment | None:
        spans = self.timeline.spans
        ftr = self.stream.ftr
        sample_rate = self.config.sample_rate
        pad_seconds = self.config.pad_seconds

        # handle masks
        words = [
            t
            for t in piece
            if self.stream.content_mask[t]
        ]
        if not words:
            return None

        aligned = [
            t
            for t in words
            if self.timeline.ok_mask[t]
        ]

        if len(aligned) / len(words) < self.config.min_aligned_fraction:
            return None

        first_span = spans[piece[0]]
        last_span = spans[piece[-1]]

        if not first_span or not last_span:
            return None

        start_s = max(0.0, first_span.start - pad_seconds)

        audio_duration = len(waveform) / sample_rate
        end_s = min(audio_duration, last_span.end + pad_seconds)

        if last_span.end <= first_span.start:
            return None

        a = round(start_s * sample_rate)
        b = round(end_s * sample_rate)

        total_score = 0
        total_count = 0
        for t in aligned:
            span = spans[t]
            if span:
                total_score += span.score
                total_count += span.count

        score = total_score / total_count

        return Segment(
            audio=waveform[a:b],
            text=" ".join(ftr.tokens[t] for t in piece),
            speaker_id=ftr.speakers[piece[0]],
            avg_word_score=score,
            start_s=start_s,
            end_s=end_s,
        )
