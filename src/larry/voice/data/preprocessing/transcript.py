import dataclasses
import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import ClassVar, Self

import torch

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry
from larry.voice.config.preprocessing.voice_configs import TranscriptConfig
from larry.voice.data.preprocessing.alignment import Aligner, SpanScore
from larry.voice.data.preprocessing.normalization import TextNormalizer


@dataclasses.dataclass
class Turn:
    """Helper dataclass for a single speaker: speech turn."""

    speaker: str
    text: str
    line_start: int
    section: int = 0


@dataclasses.dataclass
class FlattenedTranscript:
    """Helper dataclass for storing the final result of a parsed transcript as various aligned arrays."""

    tokens: list[str] = dataclasses.field(default_factory=list)
    groups: list[int] = dataclasses.field(default_factory=list)
    turn_idx: list[int] = dataclasses.field(default_factory=list)
    speakers: list[str] = dataclasses.field(default_factory=list)
    roles: list[str] = dataclasses.field(default_factory=list)
    labels: list[str] = dataclasses.field(default_factory=list)


def pdf_bytes_to_text(data: bytes) -> str:
    """Utility to parse a pdf into text from raw bytes."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext not found (apt install poppler-utils)")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        res = subprocess.run([exe, "-layout", "-enc", "UTF-8", f.name, "-"], check=True, capture_output=True)
    return res.stdout.decode("utf-8", errors="replace")


class Transcript[C: TranscriptConfig = TranscriptConfig](Registrable, ABC, root=True):
    """Superclass for transcripts, contains helpers for parsing both from raw and from already-parsed transcripts."""

    REGISTRY: ClassVar[TypeRegistry[Transcript]]

    _config: C

    turns: list[Turn]
    appearances: dict[str, str]
    reattributions: list[tuple[str, str, str]]

    raw_text: str
    body_lines: list[str]

    def __init__(self, config: C) -> None:
        """Simple init."""
        self.log = logging.getLogger(type(self).__name__)
        self._config = config
        self.turns = []
        self.appearances = {}

    @property
    def config(self) -> C:
        """Immutable type-hint getter for the config instance."""
        return self._config

    def parse_from_raw(self, raw_text: str) -> None:
        """Parses a transcript from an unnormalized source."""
        self.raw_text = raw_text
        self.body_lines = self.extract_body_lines(self.raw_text)
        self.parse_turns()

    def parse_from_turns(self, turns: list[Turn]) -> None:
        """Parses a transcript from already normalized turns."""
        self.body_lines = [f"{turn.speaker}: {turn.text}" for turn in turns]
        self.raw_text = "\n".join(self.body_lines)
        self.turns = turns

    def flatten(self) -> FlattenedTranscript:
        """Turns this ``Transcript`` instance into a ``FlattenedTranscript`` instance."""
        group_idx = 0

        ftr = FlattenedTranscript()

        for turn in self.turns:
            toks = turn.text.split()
            if not toks:
                continue

            key, role = self.resolve_speaker(turn.speaker)
            group_idx += 1
            ftr.tokens.extend(toks)
            ftr.groups.extend([group_idx] * len(toks))
            ftr.turn_idx.extend([len(ftr.labels)] * len(toks))
            ftr.speakers.extend([key] * len(toks))
            ftr.roles.extend([role] * len(toks))
            ftr.labels.append(turn.speaker)

        return ftr

    @abstractmethod
    def extract_body_lines(self, raw: str) -> list[str]:
        """Overridden by subclasses to return body lines from a raw transcript.

        Body lines are normalized lines containing speaker: speech pairs.
        """
        ...

    @abstractmethod
    def parse_turns(self) -> None:
        """Overridden by subclasses to parse body lines into speaker turns."""
        ...

    @abstractmethod
    def load_appearances(self) -> None:
        """Called once per transcript to load the bank of all possible speakers for this transcript."""
        ...

    @abstractmethod
    def resolve_speaker(self, speaker_raw: str) -> tuple[str, str]:
        """Resolves a speaker label to the canonical form."""
        ...


class NormalizedTokens:
    """Helper utility class for choosing canonical normalized tokens for readings."""

    tokens: list[str]
    readings: list[list[list[str]]]  # token -> options -> words
    choice: list[int]
    words: list[str]
    owners: list[int]
    words_by_token: list[list[str]]
    word_start: list[int]
    content_mask: list[bool]

    def __init__(self, tokens: list[str], normalizer: TextNormalizer) -> None:
        """For each input token, uses the previous token to collect possible reading variations."""
        self.tokens = tokens
        self.readings = [
            normalizer.normalize_in_context(tokens, i)
            for i in range(len(tokens))
        ]

        self.choice = [0] * len(self.readings)
        self.rebuild()

    def rebuild(self) -> None:
        """Recalculates outputs."""
        self.words = []
        self.owners = []
        self.words_by_token = []
        self.word_start = []
        for t, options in enumerate(self.readings):
            chosen = options[self.choice[t]]
            self.word_start.append(len(self.words))
            self.words_by_token.append(chosen)
            self.words.extend(chosen)
            self.owners.extend([t] * len(chosen))

        self.content_mask = [
            len(chosen) > 0
            for chosen in self.words_by_token
        ]

    def choose_readings(
            self,
            aligner: Aligner,
            emissions: torch.Tensor,
            word_spans: list[SpanScore | None],
            margin_seconds: float,
    ) -> bool:
        """Pick each ambiguous token's reading by forced-align score in its local window. True if any changed."""
        changed = False
        for t, options in enumerate(self.readings):
            if len(options) < 2:
                continue

            first = self.word_start[t]
            last = first + len(self.words_by_token[t]) - 1
            if first == 0 or last + 1 >= len(self.words):
                continue

            before = word_spans[first - 1]
            after = word_spans[last + 1]
            if before is None or after is None:
                continue

            lo = max(0, int((before.start - margin_seconds) / aligner.frame_seconds))
            hi = min(emissions.shape[0], int((after.end + margin_seconds) / aligner.frame_seconds) + 1)
            window = emissions[lo:hi]

            best = self.choice[t]
            best_score: float | None = None
            for c, option in enumerate(options):
                candidate = [self.words[first - 1]] + option + [self.words[last + 1]]
                score = aligner.path_score(window, candidate)
                if score is None:
                    continue

                if best_score is None or score > best_score:
                    best = c
                    best_score = score

            if best != self.choice[t]:
                self.choice[t] = best
                changed = True

        return changed

    def choose_readings_whole(self, aligner: Aligner, emissions: torch.Tensor) -> list[float | None]:
        """Does choose_readings' logic, but on the entire emissions for this instance."""
        results: list[float | None] = []
        for t, options in enumerate(self.readings):
            if len(options) < 2:
                continue

            best = self.choice[t]
            best_score: float | None = None
            for c in range(len(options)):
                self.choice[t] = c
                self.rebuild()
                score = aligner.path_score(emissions, self.words)
                if score is None:
                    continue

                if best_score is None or score > best_score:
                    best = c
                    best_score = score
            results.append(best_score)

            self.choice[t] = best
        self.rebuild()
        return results


class TokenStream(NormalizedTokens):
    """Container for a flattened transcript that inherits from ``NormalizedTokens`` to produce a stream of outputs."""

    ftr: FlattenedTranscript

    def __init__(self, ftr: FlattenedTranscript, normalizer: TextNormalizer) -> None:
        """Simple init."""
        self.ftr = ftr
        super().__init__(ftr.tokens, normalizer)

    @classmethod
    def from_transcript(cls, tr: Transcript, normalizer: TextNormalizer) -> Self:
        """Creates a ``TokenStream`` instance from a given ``Transcript`` and ``TextNormalizer``"""
        return cls(tr.flatten(), normalizer)

    def fold(self, spans: list[SpanScore | None]) -> list[SpanScore | None]:
        """Folds a list of spans, limiting or preventing overlaps but closing gaps."""
        acc = {}
        for s, span in enumerate(spans):
            if span is None:
                continue

            owner = self.owners[s]
            if owner in acc:
                a = acc[owner]
                a.start = min(a.start, span.start)
                a.end = max(a.end, span.end)
                a.score += span.score
                a.count += span.count
            else:
                acc[owner] = SpanScore(span.start, span.end, span.score, span.count)

        # normalized words back onto original tokens: a token spans all of its pieces.
        tok_time: list[SpanScore | None] = [None] * len(self.ftr.tokens)
        for owner, span in acc.items():
            tok_time[owner] = span

        return tok_time

    def word_turn_starts(self) -> list[int]:
        groups = self.ftr.groups
        return [
            w
            for w, token in enumerate(self.owners)
            if w == 0 or groups[token] != groups[self.owners[w - 1]]
        ]

    def fold_flags(self, word_flags: list[bool]) -> list[bool]:
        return [
            bool(words) and all(word_flags[self.word_start[t] + k] for k in range(len(words)))
            for t, words in enumerate(self.words_by_token)
        ]

    def turns(self) -> list[list[int]]:
        """Returns an inversion of the ``FlattenedTranscript`` instance's labels list."""
        by_turn: list[list[int]] = [
            []
            for _ in self.ftr.labels
        ]
        for t, turn_idx in enumerate(self.ftr.turn_idx):
            by_turn[turn_idx].append(t)

        return by_turn
