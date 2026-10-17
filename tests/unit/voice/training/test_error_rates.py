from types import SimpleNamespace

import pytest
import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from larry.common.metrics.metrics import MetricDict
from larry.common.model.model import LarrySpeechTokenizerModelOutput
from larry.common.training.trainers import StepOutput
from larry.text.tokenizer.text_tokenizer import TextTokenizer
from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.training.voice_trainers import (
    ErrorCounts,
    error_counts,
    SpeechEncoderTrainer,
    _normalize_for_scoring,
    _normalize_reference_for_scoring,
)

ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
BLANK = len(ALPHABET)
SPELLINGS = {"colour": "color", "grey": "gray"}


class CharTokenizer(TextTokenizer):
    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(ALPHABET[i] for i in token_ids)

    def vocab_size(self) -> int:
        return len(ALPHABET)

    def id_to_piece(self, token_id: int) -> str:
        return ALPHABET[token_id]


def test_identical_text_has_no_errors() -> None:
    assert error_counts("THE CASE IS SUBMITTED", "THE CASE IS SUBMITTED") == ErrorCounts(0, 4, 0, 21)


def test_word_and_character_errors_are_counted_against_the_reference() -> None:
    counts = error_counts("THE CAT SAT", "THE CAT SAT DOWN")

    assert (counts.word_errors, counts.words) == (1, 4)
    assert (counts.char_errors, counts.chars) == (5, 16)


def test_empty_hypothesis_misses_every_reference_word() -> None:
    assert error_counts("", "YES YOUR HONOR") == ErrorCounts(3, 3, 14, 14)


def test_whitespace_differences_are_not_character_errors() -> None:
    assert error_counts("  YES   SIR ", "YES SIR").char_errors == 0


def _trainer(sources: list[str] | None = None) -> SpeechEncoderTrainer:
    trainer = SpeechEncoderTrainer.__new__(SpeechEncoderTrainer)
    vars(trainer).update(
        tokenizer=CharTokenizer(),
        whisper_normalizer=EnglishTextNormalizer(SPELLINGS),
        model_config=SimpleNamespace(text_vocab_size=BLANK, fsq_dims=2, fsq_levels=3),
        _config=SimpleNamespace(mixture_config=SimpleNamespace(
            eval_sources=[SpeechSourceConfig(name=n, index_path=n) for n in sources or []],
        )),
        accelerator=SimpleNamespace(device=torch.device("cpu")),
    )
    return trainer


@pytest.mark.parametrize(("text", "expected"), [
    ("UM YES UH YOUR HONOR MM HMM", "yes your honor"),
    ("IN NINETEEN SEVENTY THE COLOUR WAS GREY", "in 1970 the color was gray"),
    ("TWENTY FIVE PERCENT OF FIVE HUNDRED DOLLARS", "25% of $500"),
    ("", ""),
])
def test_scoring_normalization_is_whispers(text: str, expected: str) -> None:
    assert _normalize_for_scoring(_trainer().whisper_normalizer, text) == expected


def test_different_readings_of_a_number_score_the_same() -> None:
    normalizer = _trainer().whisper_normalizer

    spelled = _normalize_for_scoring(normalizer, "ONE NINE SEVEN ZERO")

    assert spelled == _normalize_for_scoring(normalizer, "NINETEEN SEVENTY")


def test_reference_normalization_matches_and_is_cached() -> None:
    normalizer = _trainer().whisper_normalizer
    _normalize_reference_for_scoring.cache_clear()

    first = _normalize_reference_for_scoring(normalizer, "IN NINETEEN SEVENTY")
    second = _normalize_reference_for_scoring(normalizer, "IN NINETEEN SEVENTY")

    assert first == second == _normalize_for_scoring(normalizer, "IN NINETEEN SEVENTY")
    assert _normalize_reference_for_scoring.cache_info().hits == 1


def _frames_for(text: str) -> list[int]:
    frames = []
    for i, c in enumerate(text):
        if i > 0 and text[i - 1] == c:
            frames.append(BLANK)
        frames.append(ALPHABET.index(c))
    return frames + [BLANK]


def _log_probs(hypotheses: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    paths = [_frames_for(h) for h in hypotheses]
    length = max(len(p) for p in paths)
    scores = torch.full((len(paths), length, BLANK + 1), -10.0)
    for b, path in enumerate(paths):
        for t in range(length):
            scores[b, t, path[t] if t < len(path) else BLANK] = 0.0
    return scores.log_softmax(-1), torch.tensor([len(p) for p in paths])


def test_eval_statistics_report_greedy_error_rates_per_source_and_overall() -> None:
    trainer = _trainer(["scotus", "cv27"])
    references = ["YES YOUR HONOR", "UH A BOOK", "THANK YOU"]
    hypotheses = ["YES YOUR HONOR", "A BOOK", "THANK"]
    log_probs, frame_lengths = _log_probs(hypotheses)
    targets = torch.nn.utils.rnn.pad_sequence([torch.tensor(trainer.tokenizer.encode(r)) for r in references],
                                              batch_first=True)
    batch = {
        "targets": targets,
        "target_lengths": torch.tensor([len(r) for r in references]),
        "source_ids": torch.tensor([0, 1, 1]),
    }
    output = StepOutput(loss=torch.tensor(0.0), metrics=MetricDict(), model_output=LarrySpeechTokenizerModelOutput(
        logits=log_probs, ctc_log_probs=log_probs, frame_lengths=frame_lengths,
        token_ids=torch.zeros(frame_lengths.shape[0], log_probs.shape[1], dtype=torch.long),
    ))

    statistics = trainer.eval_statistics()
    trainer.update_eval_statistics(statistics, batch, output)
    logged = trainer.eval_statistics_metrics(statistics)
    metrics = {tag: entry.value for tag, entry in logged.items()}

    assert metrics["eval/wer/scotus"] == 0.0
    assert metrics["eval/wer/cv27"] == pytest.approx(2 / 5)
    assert metrics["eval/wer_normalized/cv27"] == pytest.approx(1 / 4)
    assert metrics["eval/wer"] == pytest.approx(2 / 8)
    assert metrics["eval/cer/cv27"] == pytest.approx((3 + 4) / (9 + 9))
