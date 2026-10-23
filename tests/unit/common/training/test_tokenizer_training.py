import collections
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import sentencepiece
import yaml

from larry.common.config.data.data_configs import TextSourceConfig
from larry.common.training import tokenizer_training
from larry.common.training.tokenizer_training import SourceText, SourceTextReader, TokenizerTrainer
from larry.text.tokenizer.source_text_readers import TextSourceTextReader
from larry.voice.config.data.voice_data_configs import ASRSourceConfig
from larry.voice.tokenizer.source_text_readers import ASRSourceTextReader
from tests.helpers.common import import_larry_submodules, write_parquet_split

ALPHABET_LINES = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS",
    "HOW VEXINGLY QUICK DAFT ZEBRAS JUMP",
    "SPHINX OF BLACK QUARTZ JUDGE MY VOW",
    "IT'S THE COURT'S DECISION",
]


def _asr_source(root: Path, name: str, rows: list[tuple[str, float]], split: str = "train") -> ASRSourceConfig:
    write_parquet_split(root / name, split, {
        "text_normalized": [t for t, _ in rows],
        "duration_s": [d for _, d in rows],
        "audio": [b"\x00" * 64 for _ in rows],
    })
    return ASRSourceConfig(name=name, index_path=str(root / name), split=split)


def _text_source(root: Path, name: str, lines: list[str], **overrides: Any) -> TextSourceConfig:
    write_parquet_split(root / name, "train", {"text": lines})
    return TextSourceConfig(name=name, index_path=str(root / name), **overrides)


def _text(name: str, weight: float, size: float, counts: dict[str, int], unit: str = "seconds") -> SourceText:
    return SourceText(name=name, weight=weight, size=size, unit=unit, counts=collections.Counter(counts))


def test_sources_resolve_to_the_reader_for_their_type_even_when_training_packages_are_skipped() -> None:
    import_larry_submodules()

    asr = SourceTextReader.REGISTRY.for_config(ASRSourceConfig(name="a", index_path="a"))
    text = SourceTextReader.REGISTRY.for_config(TextSourceConfig(name="t", index_path="t"))

    assert (asr, text) == (ASRSourceTextReader, TextSourceTextReader)


def test_a_fresh_process_importing_like_main_resolves_asr_sources_to_the_asr_reader() -> None:
    script = (
        "from larry.common.training.tokenizer_training import SourceTextReader\n"
        "from tests.helpers.common import import_larry_submodules\n"
        "from larry.voice.config.data.voice_data_configs import ASRSourceConfig\n"
        "import_larry_submodules()\n"
        "print(SourceTextReader.REGISTRY.for_config(ASRSourceConfig(name='a', index_path='a')).__name__)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "ASRSourceTextReader"


def test_asr_sources_count_lines_and_sum_seconds(tmp_path: Path) -> None:
    source = _asr_source(tmp_path, "a", [("HELLO", 1.5), ("HELLO", 2.0), ("WORLD", 0.5), ("", 9.0)])

    text = TokenizerTrainer.read_source(source)

    assert text.counts == collections.Counter({"HELLO": 2, "WORLD": 1})
    assert (text.size, text.unit, text.weight) == (pytest.approx(13.0), "seconds", 1.0)


def test_text_sources_count_lines_and_sum_characters(tmp_path: Path) -> None:
    source = _text_source(tmp_path, "t", ["Hello there.", "Hello there.", "Bye", ""], weight=0.3)

    text = TokenizerTrainer.read_source(source)

    assert text.counts == collections.Counter({"Hello there.": 2, "Bye": 1})
    assert (text.size, text.unit, text.weight) == (27.0, "characters", 0.3)


def test_a_custom_text_column_is_read(tmp_path: Path) -> None:
    write_parquet_split(tmp_path / "t", "train", {"body": ["ONE", "TWO"]})

    text = TokenizerTrainer.read_source(TextSourceConfig(name="t", index_path=str(tmp_path / "t"), text_column="body"))

    assert set(text.counts) == {"ONE", "TWO"}


def test_a_missing_split_is_reported(tmp_path: Path) -> None:
    _asr_source(tmp_path, "a", [("HELLO", 1.0)])

    with pytest.raises(FileNotFoundError, match="validation"):
        TokenizerTrainer.read_source(ASRSourceConfig(name="a", index_path=str(tmp_path / "a"), split="validation"))


def test_a_minified_index_reads_text_and_durations_from_its_source_by_row(tmp_path: Path) -> None:
    write_parquet_split(tmp_path / "materialized", "train", {
        "id": ["a", "b", "c"],
        "text_normalized": ["HELLO", "SKIPPED", "WORLD"],
        "duration_s": [1.0, 5.0, 2.5],
    })
    write_parquet_split(tmp_path / "index", "train", {"row": [0, 2], "id": ["a", "c"]})
    source = ASRSourceConfig(name="m", index_path=str(tmp_path / "index"), source_path=str(tmp_path / "materialized"))

    text = TokenizerTrainer.read_source(source)

    assert text.counts == collections.Counter({"HELLO": 1, "WORLD": 1})
    assert text.size == pytest.approx(3.5)


def test_an_index_with_text_and_durations_never_loads_its_source(tmp_path: Path) -> None:
    write_parquet_split(tmp_path / "index", "train", {
        "row": [0, 1], "id": ["a", "b"], "text_normalized": ["HI", "YO"], "duration_s": [1.0, 2.0],
    })
    source = ASRSourceConfig(name="i", index_path=str(tmp_path / "index"), source_path=str(tmp_path / "missing"))

    assert TokenizerTrainer.read_source(source).size == pytest.approx(3.0)


def test_weighted_frequencies_follow_target_shares() -> None:
    a = _text("a", weight=1.0, size=100.0, counts={"SHARED": 1, "ONLY A": 2})
    b = _text("b", weight=1.0, size=300.0, counts={"SHARED": 1, "ONLY B": 3})

    frequencies = TokenizerTrainer([], Path("sp"), 40)._weighted_frequencies([a, b])

    assert frequencies == {"ONLY A": 6, "ONLY B": 3, "SHARED": 4}


def test_text_weights_are_shares_regardless_of_source_size() -> None:
    small = _text("small", weight=1.0, size=10.0, counts={"S": 1}, unit="characters")
    large = _text("large", weight=1.0, size=1000.0, counts={"L": 100}, unit="characters")

    frequencies = TokenizerTrainer([], Path("sp"), 40)._weighted_frequencies([small, large])

    assert frequencies["S"] == frequencies["L"]


def test_sources_with_zero_weight_or_size_are_skipped() -> None:
    keep = _text("keep", weight=1.0, size=100.0, counts={"KEEP": 1})
    unweighted = _text("unweighted", weight=0.0, size=100.0, counts={"DROP": 1})
    empty = _text("empty", weight=1.0, size=0.0, counts={})

    assert TokenizerTrainer([], Path("sp"), 40)._weighted_frequencies([keep, unweighted, empty]) == {"KEEP": 1}


def test_mixtures_weighted_in_different_units_are_rejected() -> None:
    seconds = _text("speech", weight=1.0, size=10.0, counts={"A": 1})
    characters = _text("text", weight=1.0, size=10.0, counts={"B": 1}, unit="characters")

    with pytest.raises(ValueError, match="one unit"):
        TokenizerTrainer([], Path("sp"), 40)._weighted_frequencies([seconds, characters])


def test_a_mixture_with_no_positive_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        TokenizerTrainer([], Path("sp"), 40)._weighted_frequencies([_text("a", 0.0, 10.0, {"A": 1})])


@pytest.mark.parametrize("kind", ["asr", "text"])
def test_training_writes_a_model_with_the_requested_vocabulary(tmp_path: Path, kind: str) -> None:
    if kind == "asr":
        sources = [
            _asr_source(tmp_path, "a", [(line, 3.0) for line in ALPHABET_LINES] * 20),
            _asr_source(tmp_path, "b", [(line, 2.0) for line in reversed(ALPHABET_LINES)] * 10),
        ]
    else:
        sources = [_text_source(tmp_path, "t", ALPHABET_LINES * 20)]

    model_file = TokenizerTrainer(sources, tmp_path / "out" / "sp", vocab_size=40).run()

    sp = sentencepiece.SentencePieceProcessor(model_file=str(model_file))
    assert sp.get_piece_size() == 40
    assert (tmp_path / "out" / "sp.tsv").exists()
    for line in ALPHABET_LINES:
        assert sp.decode(sp.encode(line)) == line


def _write_config(tmp_path: Path, model: dict[str, Any], trainer: dict[str, Any]) -> Path:
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump({"model": model, "trainer": trainer}))
    return config_file


def _encoder_trainer(tmp_path: Path, source: ASRSourceConfig) -> dict[str, Any]:
    return {
        "type": "SpeechEncoderTrainerConfig",
        "max_steps": 1,
        "tokenizer_model_path": str(tmp_path / "tok" / "sp.model"),
        "speaker_mapping_path": str(tmp_path / "speakers.csv"),
        "mixture_config": {"train_sources": {source.name: {"index_path": source.index_path}}, "eval_sources": {}},
    }


def test_main_reads_sources_and_vocabulary_from_a_training_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _asr_source(tmp_path, "a", [(line, 3.0) for line in ALPHABET_LINES] * 20)
    config = _write_config(
        tmp_path, {"type": "SpeechEncoderModelConfig", "text_vocab_size": 40}, _encoder_trainer(tmp_path, source)
    )
    monkeypatch.setattr(sys, "argv", ["tokenizer-training", "-f", str(config), "--allow_dirty"])

    tokenizer_training.main()

    sp = sentencepiece.SentencePieceProcessor(model_file=str(tmp_path / "tok" / "sp.model"))
    assert sp.get_piece_size() == 40


def test_main_rejects_a_model_without_a_text_vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _asr_source(tmp_path, "a", [("HELLO", 1.0)])
    config = _write_config(tmp_path, {"type": "LarryVocoderConfig"}, _encoder_trainer(tmp_path, source))
    monkeypatch.setattr(sys, "argv", ["tokenizer-training", "-f", str(config), "--allow_dirty"])

    with pytest.raises(TypeError, match="text vocabulary"):
        tokenizer_training.main()


@pytest.mark.parametrize(("extra_args", "error", "match"), [
    ([], ValueError, "output_prefix"),
    (["--output_prefix", "out/sp"], TypeError, "without a text column"),
])
def test_main_needs_a_tokenizer_path_and_text_sources(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_args: list[str], error: type, match: str
) -> None:
    source = _asr_source(tmp_path, "a", [("HELLO", 1.0)])
    trainer = {
        "type": "VocoderTrainerConfig",
        "max_steps": 1,
        "mixture_config": {"train_sources": {"a": {"index_path": source.index_path}}, "eval_sources": {}},
    }
    config = _write_config(tmp_path, {"type": "SpeechEncoderModelConfig", "text_vocab_size": 40}, trainer)
    monkeypatch.setattr(sys, "argv", ["tokenizer-training", "-f", str(config), "--allow_dirty", *extra_args])

    with pytest.raises(error, match=match):
        tokenizer_training.main()
