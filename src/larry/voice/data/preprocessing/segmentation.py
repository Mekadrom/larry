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
            unspoken: list[bool] | None = None,
    ) -> None:
        """Simple init."""
        self.log = logging.getLogger(type(self).__name__)
        self.config = config
        self.stream = stream
        self.timeline = timeline
        self.unspoken = unspoken if unspoken is not None else [False] * len(stream.ftr.tokens)

    def segments(self, waveform: torch.Tensor) -> list[Segment]:
        """Splits an input waveform into segments according to the spans."""
        turns = self.stream.turns()
        crosstalk = self._crosstalk_turns(turns)
        if crosstalk:
            self.log.info(f"dropping {len(crosstalk)} crosstalk turns")

        dropped = {t for t, flag in enumerate(self.unspoken) if flag}
        pieces = []
        for turn_idx, toks in enumerate(turns):
            if turn_idx in crosstalk:
                dropped.update(toks)
                continue

            for run in self._spoken_runs(toks):
                pieces.extend(self._split_turn(run, self.timeline.spans, self.stream.ftr.tokens))

        bad = [self._is_bad(piece) for piece in pieces]
        segments = []
        for i, piece in enumerate(pieces):
            floor = ceiling = None
            if not bad[i]:
                trim_start = (i > 0 and bad[i - 1]) or piece[0] - 1 in dropped
                trim_end = (i + 1 < len(pieces) and bad[i + 1]) or piece[-1] + 1 in dropped
                piece, floor, ceiling = self._trim_edges(piece, trim_start, trim_end)

            segment = self._make_segment(piece, waveform, floor, ceiling) if piece else None
            if segment is not None:
                segments.append(segment)

        return segments

    def _spoken_runs(self, toks: list[int]) -> list[list[int]]:
        runs: list[list[int]] = [[]]
        for t in toks:
            if self.unspoken[t]:
                if runs[-1]:
                    runs.append([])
            else:
                runs[-1].append(t)

        return [
            run
            for run in runs
            if run
        ]

    def _crosstalk_turns(self, turns: list[list[int]]) -> set[int]:
        windows = self.config.crosstalk_min_turns
        bounds = [self._turn_bounds(toks) for toks in turns]
        crosstalk: set[int] = set()
        for i in range(len(bounds) - windows + 1):
            window = bounds[i:i + windows]
            if any(b is None or b[1] - b[0] >= self.config.crosstalk_max_turn_seconds for b in window):
                continue

            first, last = window[0], window[-1]
            if first is not None and last is not None and last[1] - first[0] <= self.config.crosstalk_max_seconds:
                crosstalk.update(range(i, i + windows))

        return crosstalk

    def _turn_bounds(self, toks: list[int]) -> tuple[float, float] | None:
        spans = [
            span
            for t in toks
            if not self.unspoken[t] and (span := self.timeline.spans[t]) is not None
        ]
        if not spans:
            return None

        return min(s.start for s in spans), max(s.end for s in spans)

    def _aligned(self, piece: list[int]) -> list[int]:
        aligned = []
        for t in piece:
            if not self.stream.content_mask[t] or not self.timeline.ok_mask[t]:
                continue

            span = self.timeline.spans[t]
            if span is None:
                raise ValueError(f"null span: t={t}")

            if span.end - span.start <= self.config.max_word_seconds:
                aligned.append(t)

        return aligned

    def _score(self, tokens: list[int]) -> float | None:
        spans = [span for t in tokens if (span := self.timeline.spans[t]) is not None]
        count = sum(span.count for span in spans)
        if count == 0:
            return None

        return sum(span.score for span in spans) / count

    def _is_bad(self, piece: list[int]) -> bool:
        score = self._score(self._aligned(piece))
        return score is None or score < self.config.bad_segment_score

    def _trim_edges(
            self,
            piece: list[int],
            trim_start: bool,
            trim_end: bool,
    ) -> tuple[list[int], float | None, float | None]:
        start, end = 0, len(piece)
        while trim_start and start < end and not self._edge_ok(piece[start]):
            start += 1

        while trim_end and end > start and not self._edge_ok(piece[end - 1]):
            end -= 1

        if start > 0:
            floor = self._span_edge(piece[start - 1], "end")
        else:
            floor = None

        if end < len(piece):
            ceiling = self._span_edge(piece[end], "start")
        else:
            ceiling = None

        return piece[start:end], floor, ceiling

    def _span_edge(self, t: int, edge: str) -> float | None:
        span = self.timeline.spans[t]
        if span is None or not self.timeline.ok_mask[t]:
            return None

        return span.end if edge == "end" else span.start

    def _edge_ok(self, t: int) -> bool:
        if not self.stream.content_mask[t] or not self.timeline.ok_mask[t]:
            return False

        score = self._score([t])
        return score is not None and score >= self.config.edge_word_min_score

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

    def _make_segment(
            self,
            piece: list[int],
            waveform: torch.Tensor,
            floor: float | None = None,
            ceiling: float | None = None,
    ) -> Segment | None:
        """Produces a single segment from a set of aligned words in sequence."""
        spans = self.timeline.spans
        ftr = self.stream.ftr
        sample_rate = self.config.sample_rate
        pad_start_seconds = self.config.pad_start_seconds
        pad_end_seconds = self.config.pad_end_seconds

        # handle masks
        words = [
            t
            for t in piece
            if self.stream.content_mask[t]
        ]
        if not words:
            return None

        aligned = self._aligned(piece)

        if not aligned or len(aligned) / len(words) < self.config.min_aligned_fraction:
            return None

        first_span = spans[piece[0]]
        last_span = spans[piece[-1]]
        nxt = piece[-1] + 1

        if not first_span or not last_span:
            return None

        start_s = max(0.0, first_span.start - pad_start_seconds, floor if floor is not None else 0.0)

        audio_duration = len(waveform) / sample_rate
        limit = audio_duration
        if nxt < len(spans):
            next_span = spans[nxt]
            if next_span is not None:
                limit = next_span.start

        end_s = min(audio_duration, last_span.end + pad_end_seconds, max(last_span.end + pad_start_seconds, limit))
        if ceiling is not None:
            end_s = min(end_s, max(ceiling, last_span.end))

        length = end_s - start_s
        if length > self.config.max_segment_seconds:
            self.log.warning(f"dropping {length:.1f}s segment at {start_s:.1f}s: exceeds cap")
            return None

        if last_span.end <= first_span.start:
            return None

        a = round(start_s * sample_rate)
        b = round(end_s * sample_rate)

        score = self._score(aligned)

        text = " ".join(ftr.tokens[t] for t in piece)
        if score is None:
            raise ValueError(f"Bad score alignment made it past _is_bad check: {text}")

        return Segment(
            audio=waveform[a:b],
            text=text,
            text_normalized=" ".join(w for t in piece for w in self.stream.words_by_token[t]),
            speaker_id=ftr.speakers[piece[0]],
            avg_word_score=score,
            start_s=start_s,
            end_s=end_s,
        )
