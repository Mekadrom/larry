import dataclasses
import logging

import torch

from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig
from larry.voice.data.preprocessing.alignment import SpanScore, TokenTimeline
from larry.voice.data.preprocessing.normalization import ABBREVIATIONS
from larry.voice.data.preprocessing.transcript import TokenStream


@dataclasses.dataclass
class Segment:
    """Helper dataclass for storing all data relevant to a single segment of speech."""

    audio: torch.Tensor
    text: str
    text_normalized: str
    speaker_id: str
    avg_word_score: float
    start_s: float
    end_s: float


class Segmenter:
    """Utility class for splitting a larger speech and transcript into individual audio files and speaker turns."""

    def __init__(
            self,
            config: SpeechSegmentationAligningMapperConfig,
            stream: TokenStream,
            timeline: TokenTimeline,
    ) -> None:
        """Simple init."""
        self.log = logging.getLogger(type(self).__name__)
        self.config = config
        self.stream = stream
        self.timeline = timeline

    def segments(self, waveform: torch.Tensor) -> list[Segment]:
        """Splits an input waveform into segments according to the spans."""
        segments = []
        for toks in self.stream.turns():
            for piece in self._split_turn(toks, self.timeline.spans, self.stream.ftr.tokens):
                segment = self._make_segment(piece, waveform)
                if segment is not None:
                    segments.append(segment)

        return segments

    def _split_turn(self, toks: list[int], spans: list[SpanScore | None], tokens: list[str]) -> list[list[int]]:
        """Split a turn into pieces under max_segment_seconds, preferring to split on sentence ends."""
        pieces = []
        cur = []
        prev = None
        cap = self.config.max_segment_seconds - self.config.pad_start_seconds - self.config.pad_end_seconds
        for t in toks:
            cur_span = spans[t]
            if cur_span is None:
                continue

            gap = 0.0
            if prev is not None:
                prev_span = spans[prev]
                if prev_span is not None:
                    gap = cur_span.start - prev_span.end

            pause = self.config.max_pause_seconds is not None and gap > self.config.max_pause_seconds
            too_long = bool(cur) and cur_span.end - spans[cur[0]].start > cap

            if cur and (pause or too_long):
                stops = [
                    k
                    for k, c in enumerate(cur)
                    if self._is_sentence_end(tokens[c])
                ]

                cut = len(cur) if pause or not stops else stops[-1] + 1

                # sentence cut leaves a remainder that's still too long with t; cut everything instead
                if cut < len(cur) and cur_span.end - spans[cur[cut]].start > cap:
                    cut = len(cur)

                pieces.append(cur[:cut])
                cur = cur[cut:]

            cur.append(t)
            prev = t

        if cur:
            pieces.append(cur)

        return pieces

    def _is_sentence_end(self, token: str) -> bool:
        """Detects the end of a sentence from the provided token."""
        core = token.rstrip("\"')")
        if not core.endswith((".", "?", "!")):
            return False

        if core.endswith("."):
            base = core.lstrip("(\"'").rstrip(".").upper()
            if base in ABBREVIATIONS:
                return False

            # a single-letter initial: "John J. Smith", "D. John Sauer"
            if len(base) == 1 and base.isalpha():
                return False

        return True

    def _make_segment(self, piece: list[int], waveform: torch.Tensor) -> Segment | None:
        """Produces a single segment from a set of aligned words in sequence."""
        spans = self.timeline.spans
        ftr = self.stream.ftr
        sample_rate = self.config.sample_rate
        pad_start_seconds = self.config.pad_start_seconds
        pad_end_seconds = self.config.pad_end_seconds
        max_word_seconds = self.config.max_word_seconds

        # handle masks
        words = [
            t
            for t in piece
            if self.stream.content_mask[t]
        ]
        if not words:
            return None

        aligned = []
        for t in words:
            if self.timeline.ok_mask[t]:
                span = spans[t]
                if span is not None:
                    if span.end - span.start <= max_word_seconds:
                        aligned.append(t)
                else:
                    raise ValueError(f"null span: t={t}")

        if len(aligned) / len(words) < self.config.min_aligned_fraction:
            return None

        first_span = spans[piece[0]]
        last_span = spans[piece[-1]]
        nxt = piece[-1] + 1

        if not first_span or not last_span:
            return None

        start_s = max(0.0, first_span.start - pad_start_seconds)

        audio_duration = len(waveform) / sample_rate
        limit = audio_duration
        if nxt < len(spans):
            next_span = spans[nxt]
            if next_span is not None:
                limit = next_span.start
        end_s = min(audio_duration, last_span.end + pad_end_seconds, max(last_span.end + pad_start_seconds, limit))

        length = end_s - start_s
        if length > self.config.max_segment_seconds:
            self.log.warning(f"dropping {length:.1f}s segment at {start_s:.1f}s: exceeds cap")
            return None

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
            text_normalized=" ".join(w for t in piece for w in self.stream.words_by_token[t]),
            speaker_id=ftr.speakers[piece[0]],
            avg_word_score=score,
            start_s=start_s,
            end_s=end_s,
        )
