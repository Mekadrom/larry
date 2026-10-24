from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import sentencepiece
import torch
from datasets import Audio, Dataset, Value

from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer
from larry.voice.config.preprocessing.voice_mapper_configs import LarrySpeechEncoderASRMapperConfig
from larry.voice.data.preprocessing import voice_mappers
from larry.voice.data.preprocessing.voice_mappers import LarrySpeechEncoderASRMapper
from larry.voice.model.speech_encoder import SpeechEncoderModel
from tests.helpers.audio import wav
from tests.helpers.common import provenance
from tests.helpers.voice import LETTERS, CharTokenizer, tiny_speech_encoder_config

BLANK = len(LETTERS)


class StubEncoder:
    def __init__(self, frames: list[list[int]], frame_lengths: list[int]) -> None:
        self.config = SimpleNamespace(vocab_size=BLANK)
        self.frames = frames
        self.frame_lengths = frame_lengths
        self.calls: list[tuple[int, ...]] = []

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter([torch.zeros(1)])

    def __call__(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> SimpleNamespace:
        self.calls.append(tuple(waveforms.shape))
        logits = torch.full((len(self.frames), max(map(len, self.frames)), BLANK + 1), -10.0)
        for i, ids in enumerate(self.frames):
            for t, token in enumerate(ids):
                logits[i, t, token] = 0.0
        return SimpleNamespace(logits=logits.log_softmax(-1), frame_lengths=torch.tensor(self.frame_lengths))


def _ids(text: str) -> list[int]:
    return [LETTERS.index(c) for c in text]


def _audio(seconds: float) -> dict[str, object]:
    return {"bytes": wav(seconds, 16_000), "path": None}


def _mapper(**overrides: object) -> LarrySpeechEncoderASRMapper:
    config = LarrySpeechEncoderASRMapperConfig(**{
        "speech_encoder_checkpoint_path": "encoder",
        "text_tokenizer_path": "tokenizer.model",
        "num_proc": 1,
        "cache_results": False,
        **overrides,
    })
    return LarrySpeechEncoderASRMapper(provenance(), config)


Install = Callable[[list[list[int]], list[int]], StubEncoder]


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Install:
    def install(frames: list[list[int]], frame_lengths: list[int]) -> StubEncoder:
        encoder = StubEncoder(frames, frame_lengths)
        monkeypatch.setitem(voice_mappers._ENCODERS, "encoder", encoder)
        monkeypatch.setitem(voice_mappers._TOKENIZERS, "tokenizer.model", CharTokenizer())
        return encoder

    return install


def test_columns_follow_the_config() -> None:
    default = LarrySpeechEncoderASRMapperConfig(speech_encoder_checkpoint_path="e", text_tokenizer_path="t")
    renamed = LarrySpeechEncoderASRMapperConfig(
        speech_encoder_checkpoint_path="e", text_tokenizer_path="t", input_column="clip", output_column="hyp"
    )

    assert (default.input_columns, default.output_columns) == (["audio"], ["text_normalized"])
    assert (renamed.input_columns, renamed.output_columns) == (["clip"], ["hyp"])


def test_greedy_ctc_collapses_repeats_and_removes_blanks(stub: Install) -> None:
    stub([_ids("HH") + [BLANK] + _ids("EE") + [BLANK, BLANK] + _ids("LL") + [BLANK] + _ids("LO")], [12])

    out = _mapper().preprocess_batch({"audio": [_audio(0.5)]})

    assert out == {"text_normalized": ["HELLO"]}


def test_frames_past_each_examples_length_are_ignored(stub: Install) -> None:
    stub([_ids("HI") + _ids("XYZ"), _ids("OK") + [BLANK] + _ids("Q") + _ids("ZZ")], [2, 4])

    out = _mapper().preprocess_batch({"audio": [_audio(0.5), _audio(0.3)]})

    assert out["text_normalized"] == ["HI", "OKQ"]


def test_audio_is_read_from_the_configured_input_column(stub: Install) -> None:
    encoder = stub([_ids("A"), _ids("B")], [1, 1])

    out = _mapper(input_column="clip", output_column="hyp").preprocess_batch({"clip": [_audio(1.0), _audio(0.5)]})

    assert out == {"hyp": ["A", "B"]}
    assert encoder.calls == [(2, 16_000)]


def test_provenance_records_the_encoder_and_tokenizer() -> None:
    mapper = _mapper()

    assert {"encoder", "tokenizer.model"} <= mapper.provenance.ancestry.models


def test_output_feature_is_a_string_column() -> None:
    assert _mapper().output_features(Dataset.from_dict({"audio": []}))["text_normalized"] == Value("string")


def _sentencepiece(tmp_path: Path) -> Path:
    lines = ["THE QUICK BROWN FOX", "JUMPS OVER THE LAZY DOG", "PACK MY BOX WITH FIVE DOZEN JUGS"] * 30
    (tmp_path / "text.txt").write_text("\n".join(lines))
    sentencepiece.SentencePieceTrainer.train(
        input=str(tmp_path / "text.txt"),
        model_prefix=str(tmp_path / "sp"),
        vocab_size=40,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    return tmp_path / "sp.model"


def test_the_tokenizer_is_loaded_once_from_a_sentencepiece_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = str(_sentencepiece(tmp_path))
    monkeypatch.setattr(voice_mappers, "_TOKENIZERS", {})

    first = voice_mappers._lazy_get_tokenizer(path)

    assert isinstance(first, SentencePieceTokenizer)
    assert voice_mappers._lazy_get_tokenizer(path) is first
    assert first.vocab_size() == 40


@pytest.fixture
def real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    tokenizer = str(_sentencepiece(tmp_path))
    torch.manual_seed(0)
    encoder = tmp_path / "encoder"
    SpeechEncoderModel(tiny_speech_encoder_config(40)).save_pretrained(encoder)
    monkeypatch.setattr(voice_mappers, "_TOKENIZERS", {})
    monkeypatch.setitem(voice_mappers._ENCODERS, str(encoder), SpeechEncoderModel.from_pretrained(encoder))
    return str(encoder), tokenizer


def test_mapping_a_dataset_with_a_real_encoder_adds_text_and_keeps_rows(real: tuple[str, str]) -> None:
    encoder, tokenizer = real
    clips = [_audio(1.2), _audio(0.4), _audio(0.8)]
    dataset = Dataset.from_dict({"id": ["a", "b", "c"], "audio": clips}).cast_column("audio", Audio(decode=False))
    mapper = _mapper(speech_encoder_checkpoint_path=encoder, text_tokenizer_path=tokenizer, num_proc=None)

    out = mapper.preprocess_dataset(dataset)

    assert out["id"] == ["a", "b", "c"]
    assert out.features["text_normalized"] == Value("string")
    assert all(isinstance(text, str) for text in out["text_normalized"])


def test_transcripts_do_not_depend_on_the_rest_of_the_batch(real: tuple[str, str]) -> None:
    encoder, tokenizer = real
    mapper = _mapper(speech_encoder_checkpoint_path=encoder, text_tokenizer_path=tokenizer)
    clips = [_audio(1.5), _audio(0.4), _audio(0.9)]

    batched = mapper.preprocess_batch({"audio": clips})["text_normalized"]
    alone = [mapper.preprocess_batch({"audio": [clip]})["text_normalized"][0] for clip in clips]

    assert batched == alone
