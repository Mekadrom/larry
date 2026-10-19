from typing import Any

import pytest

from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig
from larry.voice.data.preprocessing.alignment import LongAlignment
from tests.helpers.alignment import DELIM, VOCAB, SyntheticSpeech, synthetic_aligner

SPOKEN = (
    "WE WILL HEAR ARGUMENT IN SOSSAMON VERSUS TEXAS YOU DONT LOOK LIKE MISTER SCHNAPPER MISTER RUSSELL "
    "BY ACCEPTING FEDERAL FUNDS FOR ITS PRISONS TEXAS CONSENTED TO SUIT FOR APPROPRIATE RELIEF"
)
UNSPOKEN = "WE WILL HEAR ARGUMENT NEXT IN CASE STAUB PROCTOR HOSPITAL ACTUALLY ITS RUSSELL OH IM SORRY"


def _config(**overrides: Any) -> SpeechSegmentationAligningMapperConfig:
    return SpeechSegmentationAligningMapperConfig(**{"block_seconds": 1.0, "slack_seconds": 0.0, **overrides})


def _run(speech: SyntheticSpeech, transcript: str, **overrides: Any) -> Any:
    return LongAlignment(_config(**overrides), synthetic_aligner(), speech.emissions(), transcript.split()).run()


def _start_frame(span: Any) -> int:
    return round(span.start / 0.02)


def test_a_matching_transcript_has_nothing_unspoken_and_aligns_every_word() -> None:
    speech = SyntheticSpeech().silence(5).say(SPOKEN).silence(5)

    result = _run(speech, SPOKEN)

    assert not any(result.unspoken)
    assert all(span is not None for span in result.spans)


def test_transcript_text_missing_from_the_start_of_the_audio_is_unspoken() -> None:
    speech = SyntheticSpeech().silence(3).say(SPOKEN).silence(5)
    n_unspoken = len(UNSPOKEN.split())

    result = _run(speech, f"{UNSPOKEN} {SPOKEN}")

    assert result.unspoken == [True] * n_unspoken + [False] * len(SPOKEN.split())
    assert all(span is None for span in result.spans[:n_unspoken])
    by_word = result.spans[n_unspoken:]
    for word_index, (word, start, _) in enumerate(speech.word_frames):
        span = by_word[word_index]
        assert span is not None, word
        assert abs(_start_frame(span) - start) <= 2, word


def test_transcript_text_missing_from_the_end_of_the_audio_is_unspoken() -> None:
    speech = SyntheticSpeech().silence(3).say(SPOKEN).silence(3)
    tail = "THANK YOU COUNSEL THE CASE IS SUBMITTED AND WE ARE ADJOURNED FOR THE DAY"

    result = _run(speech, f"{SPOKEN} {tail}")

    assert result.unspoken == [False] * len(SPOKEN.split()) + [True] * len(tail.split())


def test_with_detection_off_the_missing_text_is_forced_into_the_audio() -> None:
    speech = SyntheticSpeech().silence(3).say(SPOKEN).silence(5)

    result = _run(speech, f"{UNSPOKEN} {SPOKEN}", detect_unspoken=False)

    assert not any(result.unspoken)
    assert result.spans[0] is not None


def test_missing_text_is_found_from_short_matches_when_the_hypothesis_is_noisy() -> None:
    misheard = {2: "HERE", 5: "SACEMAN", 9: "HE", 13: "SHAPAN", 17: "FEDERALS", 21: "PRISON"}
    heard = " ".join(misheard.get(k, word) for k, word in enumerate(SPOKEN.split()))
    speech = SyntheticSpeech().silence(3).say(heard).silence(5)
    n_unspoken = len(UNSPOKEN.split())

    result = _run(speech, f"{UNSPOKEN} {SPOKEN}", min_unspoken_anchor_run=10_000)

    assert result.unspoken == [True] * n_unspoken + [False] * len(SPOKEN.split())


def test_more_unspoken_text_than_allowed_is_left_for_forced_alignment() -> None:
    speech = SyntheticSpeech().silence(3).say(SPOKEN).silence(5)

    result = _run(speech, f"{UNSPOKEN} {SPOKEN}", max_unspoken_words=3)

    assert not any(result.unspoken)


def test_wildcards_are_inserted_before_the_requested_words_and_after_the_last() -> None:
    aligner = synthetic_aligner()
    wildcard = len(VOCAB)

    targets, owners = aligner.get_targets_owners(["AB", "C"], wildcards=[0, 1, 2], wildcard=wildcard)

    a, b, c = VOCAB["A"], VOCAB["B"], VOCAB["C"]
    assert targets == [wildcard, a, b, DELIM, wildcard, c, DELIM, wildcard]
    assert owners == [-1, 0, 0, -1, -1, 1, -1, -1]


def test_untranscribed_speech_between_turns_is_absorbed_by_a_wildcard() -> None:
    speech = SyntheticSpeech().silence(2).say("ALPHA").say("MAY IT PLEASE THE COURT").say("BRAVO").silence(2)
    aligner = synthetic_aligner()

    spans = aligner.align_block(speech.emissions(), ["ALPHA", "BRAVO"], wildcards=[1])

    for word, span in zip(["ALPHA", "BRAVO"], spans, strict=True):
        start, end = speech.frames_of(word)
        assert span is not None
        assert abs(span.start - start) <= 1 and abs(span.end - (end - 2)) <= 1, word


def test_a_block_too_short_for_its_wildcards_falls_back_to_plain_alignment() -> None:
    speech = SyntheticSpeech(frames_per_char=1).say("AB")
    emissions = speech.emissions()[:3]
    aligner = synthetic_aligner()

    spans = aligner.align_block(emissions, ["AB"], wildcards=[0, 1])

    assert spans[0] is not None


@pytest.mark.parametrize("penalty", [0.5, 1.0, 2.0])
def test_wildcard_scores_trail_the_best_token_by_the_penalty(penalty: float) -> None:
    emissions = SyntheticSpeech().say("AB").emissions()
    aligner = synthetic_aligner(wildcard_penalty=penalty)

    wildcard = aligner.wildcard_emissions(emissions)

    assert wildcard.allclose(emissions.max(dim=-1).values - penalty)


def test_audio_past_the_last_word_of_a_block_goes_to_the_edge_wildcard() -> None:
    speech = SyntheticSpeech().silence(2).say("RUSSELL BY").say("ACCEPTING FEDERAL FUNDS").silence(2)
    aligner = synthetic_aligner()

    spans = aligner.align_block(speech.emissions(), ["RUSSELL", "BY"], wildcards=[0, 1, 2])

    by = spans[1]
    start, end = speech.frames_of("BY")
    assert by is not None
    assert abs(by.start - start) <= 1 and abs(by.end - (end - 2)) <= 1
