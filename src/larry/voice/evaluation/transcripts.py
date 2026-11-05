import dataclasses
import functools
import json

from huggingface_hub import hf_hub_download
from rapidfuzz.distance import Levenshtein
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

WHISPER_NORMALIZER_REPO = "openai/whisper-small"
WHISPER_NORMALIZER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"


@dataclasses.dataclass
class ErrorCounts:
    """Helper dataclass for containing the results of an evaluation from the configured whisper model."""

    word_errors: int
    words: int
    char_errors: int
    chars: int

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            word_errors=self.word_errors + other.word_errors,
            words=self.words + other.words,
            char_errors=self.char_errors + other.char_errors,
            chars=self.chars + other.chars,
        )

    @property
    def wer(self) -> float | None:
        return self.word_errors / self.words if self.words else None

    @property
    def cer(self) -> float | None:
        return self.char_errors / self.chars if self.chars else None


def error_counts(hypothesis: str, reference: str) -> ErrorCounts:
    """Word and character edit distances of ``hypothesis`` against ``reference``, with the reference lengths."""
    reference_words = reference.split()
    reference_chars = " ".join(reference_words)
    return ErrorCounts(
        word_errors=Levenshtein.distance(hypothesis.split(), reference_words),
        words=len(reference_words),
        char_errors=Levenshtein.distance(" ".join(hypothesis.split()), reference_chars),
        chars=len(reference_chars),
    )


@functools.cache
def load_whisper_normalizer() -> EnglishTextNormalizer:
    path = hf_hub_download(WHISPER_NORMALIZER_REPO, "normalizer.json", revision=WHISPER_NORMALIZER_REVISION)
    with open(path, encoding="utf-8") as f:
        return EnglishTextNormalizer(json.load(f))


def normalize_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """Whisper normalization: lowercase, fillers dropped, numbers as digits, American spellings, no punctuation."""
    return whisper_normalizer(text)


@functools.cache
def normalize_reference_for_scoring(whisper_normalizer: EnglishTextNormalizer, text: str) -> str:
    """``normalize_for_scoring`` for references, which repeat at every eval."""
    return normalize_for_scoring(whisper_normalizer, text)
