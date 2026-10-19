import logging
import math
import string

import torch

from larry.voice.config.preprocessing.voice_configs import AlignerConfig
from larry.voice.data.preprocessing.alignment import Aligner
from tests.helpers.common import uninitialized

BLANK = 0
DELIM = 1
VOCAB = {"<pad>": BLANK, "|": DELIM, **{c: i + 2 for i, c in enumerate(string.ascii_uppercase + "'")}}


def synthetic_aligner(**config: object) -> Aligner:
    aligner_config = AlignerConfig(**config)
    return uninitialized(
        Aligner,
        log=logging.getLogger("synthetic-aligner"),
        config=aligner_config,
        frame_seconds=aligner_config.stride / aligner_config.sample_rate,
        vocab=VOCAB,
        inv_vocab={v: k for k, v in VOCAB.items()},
        blank=BLANK,
        delim=DELIM,
    )


class SyntheticSpeech:
    def __init__(self, frames_per_char: int = 2, peak: float = 0.9) -> None:
        self.frames_per_char = frames_per_char
        self.peak = peak
        self.labels: list[int] = []
        self.word_frames: list[tuple[str, int, int]] = []

    def say(self, text: str) -> SyntheticSpeech:
        for word in text.split():
            start = len(self.labels)
            for c in word:
                self.labels.extend([VOCAB[c]] * self.frames_per_char + [BLANK])
            self.word_frames.append((word, start, len(self.labels)))
            self.labels.extend([DELIM, BLANK, BLANK])
        return self

    def silence(self, frames: int) -> SyntheticSpeech:
        self.labels.extend([BLANK] * frames)
        return self

    def emissions(self) -> torch.Tensor:
        n_classes = len(VOCAB)
        off = math.log((1 - self.peak) / (n_classes - 1))
        out = torch.full((len(self.labels), n_classes), off)
        out[torch.arange(len(self.labels)), torch.tensor(self.labels)] = math.log(self.peak)
        return out

    def frames_of(self, word: str, occurrence: int = 0) -> tuple[int, int]:
        matches = [(start, end) for w, start, end in self.word_frames if w == word]
        return matches[occurrence]
