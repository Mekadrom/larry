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
from larry.voice.data.preprocessing.alignment import SpanScore, Aligner
from larry.voice.data.preprocessing.normalization import TextNormalizer


@dataclasses.dataclass
class Turn:
    speaker: str
    text: str
    line_start: int


@dataclasses.dataclass
class FlattenedTranscript:
    tokens: list[str] = dataclasses.field(default_factory=list)
    groups: list[int] = dataclasses.field(default_factory=list)
    turn_idx: list[int] = dataclasses.field(default_factory=list)
    speakers: list[str] = dataclasses.field(default_factory=list)
    roles: list[str] = dataclasses.field(default_factory=list)
    labels: list[str] = dataclasses.field(default_factory=list)


def pdf_bytes_to_text(data: bytes) -> str:
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext not found (apt install poppler-utils)")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        res = subprocess.run([exe, "-layout", "-enc", "UTF-8", f.name, "-"], check=True, capture_output=True)
    return res.stdout.decode("utf-8", errors="replace")


class Transcript[C: TranscriptConfig = TranscriptConfig](Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[Transcript]]

    _config: C

    turns: list[Turn]
    appearances: dict[str, str]

    raw_text: str
    body_lines: list[str]

    def __init__(self, config: C) -> None:
        self.log = logging.getLogger(type(self).__name__)
        self._config = config
        self.turns = []
        self.appearances = {}

    @property
    def config(self) -> C:
        return self._config

    def parse_from_raw(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.body_lines = self.extract_body_lines(self.raw_text)
        self.parse_turns()

    def parse_from_turns(self, turns: list[Turn]) -> None:
        self.body_lines = [f"{turn.speaker}: {turn.text}" for turn in turns]
        self.raw_text = "\n".join(self.body_lines)
        self.turns = turns

    def flatten(self) -> FlattenedTranscript:
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
        ...

    @abstractmethod
    def parse_turns(self) -> None:
        ...

    @abstractmethod
    def load_appearances(self) -> None:
        ...

    @abstractmethod
    def resolve_speaker(self, speaker_raw: str) -> tuple[str, str]:
        ...


class NormalizedTokens:
    tokens: list[str]
    readings: list[list[list[str]]]  # token -> options -> words
    choice: list[int]
    words: list[str]
    owners: list[int]
    words_by_token: list[list[str]]
    word_start: list[int]
    content_mask: list[bool]

    def __init__(self, tokens: list[str], normalizer: TextNormalizer) -> None:
        self.tokens = tokens
        self.readings = [
            normalizer.normalize_token(token)
            for token in tokens
        ]
        self.choice = [0] * len(self.readings)
        self.rebuild()

    def rebuild(self) -> None:
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

    def choose_readings_whole(self, aligner: Aligner, emissions: torch.Tensor) -> None:
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

            self.choice[t] = best
        self.rebuild()


class TokenStream(NormalizedTokens):
    ftr: FlattenedTranscript

    def __init__(self, ftr: FlattenedTranscript, normalizer: TextNormalizer) -> None:
        self.ftr = ftr
        super().__init__(ftr.tokens, normalizer)

    @classmethod
    def from_transcript(cls, tr: Transcript, normalizer: TextNormalizer) -> Self:
        return cls(tr.flatten(), normalizer)

    def fold(self, spans: list[SpanScore | None]) -> list[SpanScore | None]:
        acc = {}
        for s, span in enumerate(spans):
            if span is None:
                continue

            owner = self.owners[s]
            if owner in acc:
                a = acc[owner]
                a.start, a.end = min(a.start, span.start), max(a.end, span.end)
                a.score += span.score
                a.count += span.count
            else:
                acc[owner] = SpanScore(span.start, span.end, span.score, span.count)

        # normalized words back onto original tokens: a token spans all of its pieces.
        tok_time: list[SpanScore | None] = [None] * len(self.ftr.tokens)
        for owner, span in acc.items():
            tok_time[owner] = span

        return tok_time

    def turns(self) -> list[list[int]]:
        by_turn: list[list[int]] = [[] for _ in self.ftr.labels]
        for t, turn_idx in enumerate(self.ftr.turn_idx):
            by_turn[turn_idx].append(t)
        return by_turn
