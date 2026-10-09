import dataclasses
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import ClassVar, Self

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry
from larry.voice.config.preprocessing.voice_configs import TranscriptConfig
from larry.voice.data.preprocessing.alignment import SpanScore
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


class TokenStream:
    ftr: FlattenedTranscript
    words: list[str]
    words_by_token: list[list[str]]
    owners: list[int]
    content_mask: list[bool]

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

        self.content_mask = [False] * len(self.ftr.tokens)
        for o in self.owners:
            self.content_mask[o] = True

        return tok_time

    def turns(self) -> list[list[int]]:
        by_turn: list[list[int]] = [[] for _ in self.ftr.labels]
        for t, turn_idx in enumerate(self.ftr.turn_idx):
            by_turn[turn_idx].append(t)
        return by_turn

    @classmethod
    def from_transcript(cls, tr: Transcript, normalizer: TextNormalizer) -> Self:
        token_stream = cls()
        token_stream.ftr = tr.flatten()
        token_stream.words, token_stream.owners = normalizer.normalize_stream(token_stream.ftr.tokens)
        token_stream.words_by_token = [[] for _ in token_stream.ftr.tokens]
        for w, o in zip(token_stream.words, token_stream.owners):
            token_stream.words_by_token[o].append(w)
        return token_stream
