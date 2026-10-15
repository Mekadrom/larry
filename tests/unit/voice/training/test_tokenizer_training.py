import collections
import sys
from pathlib import Path

import pytest
import sentencepiece
import yaml
from datasets import Dataset

from larry.voice.config.preprocessing.voice_configs import SpeechSourceConfig
from larry.voice.training import tokenizer_training
from larry.voice.training.tokenizer_training import SourceText, read_source, train_tokenizer, weighted_frequencies

ALPHABET_LINES = [
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS",
    "HOW VEXINGLY QUICK DAFT ZEBRAS JUMP",
    "SPHINX OF BLACK QUARTZ JUDGE MY VOW",
    "IT'S THE COURT'S DECISION",
]


def _write_source(root: Path, name: str, rows: list[tuple[str, float]], split: str = "train") -> SpeechSourceConfig:
    (root / name / split).mkdir(parents=True)
    Dataset.from_dict({
        "text_normalized": [t for t, _ in rows],
        "duration_s": [d for _, d in rows],
        "audio": [b"\x00" * 64 for _ in rows],
    }).to_parquet(root / name / split / f"{split}-00000-of-00001.parquet")
    return SpeechSourceConfig(name=name, index_path=str(root / name), split=split)


def test_read_source_counts_lines_and_sums_seconds(tmp_path: Path) -> None:
    source = _write_source(tmp_path, "a", [("HELLO", 1.5), ("HELLO", 2.0), ("WORLD", 0.5), ("", 9.0)])

    text = read_source(source)

    assert text.counts == collections.Counter({"HELLO": 2, "WORLD": 1})
    assert text.seconds == pytest.approx(13.0)


def test_read_source_reports_a_missing_split(tmp_path: Path) -> None:
    _write_source(tmp_path, "a", [("HELLO", 1.0)])

    with pytest.raises(FileNotFoundError, match="validation"):
        read_source(SpeechSourceConfig(name="a", index_path=str(tmp_path / "a"), split="validation"))


def test_weighted_frequencies_follow_time_share_targets() -> None:
    a = SourceText("a", weight=1.0, seconds=100.0, counts=collections.Counter({"SHARED": 1, "ONLY A": 2}))
    b = SourceText("b", weight=1.0, seconds=300.0, counts=collections.Counter({"SHARED": 1, "ONLY B": 3}))

    frequencies = weighted_frequencies([a, b])

    assert frequencies == {"ONLY A": 6, "ONLY B": 3, "SHARED": 4}


def test_weighted_frequencies_skip_sources_with_zero_weight() -> None:
    a = SourceText("a", weight=1.0, seconds=100.0, counts=collections.Counter({"KEEP": 1}))
    b = SourceText("b", weight=0.0, seconds=100.0, counts=collections.Counter({"DROP": 1}))

    assert weighted_frequencies([a, b]) == {"KEEP": 1}


def test_train_tokenizer_writes_a_model_with_the_requested_vocabulary(tmp_path: Path) -> None:
    sources = [
        _write_source(tmp_path, "a", [(line, 3.0) for line in ALPHABET_LINES] * 20),
        _write_source(tmp_path, "b", [(line, 2.0) for line in reversed(ALPHABET_LINES)] * 10),
    ]

    model_file = train_tokenizer(sources, tmp_path / "out" / "sp", vocab_size=40)

    sp = sentencepiece.SentencePieceProcessor(model_file=str(model_file))
    assert sp.get_piece_size() == 40
    assert (tmp_path / "out" / "sp.tsv").exists()
    for line in ALPHABET_LINES:
        assert sp.decode(sp.encode(line)) == line


def test_main_reads_sources_and_vocabulary_from_a_training_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path, "a", [(line, 3.0) for line in ALPHABET_LINES] * 20)
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump({
        "model": {
            "name": "SpeechTokenizerModel",
            "config": "SpeechTokenizerModelConfig",
            "overrides": [{"name": "text_vocab_size", "value": 40}],
        },
        "trainer": {
            "name": "SpeechTokenizerTrainer",
            "config": "SpeechTokenizerTrainerConfig",
            "overrides": {
                "max_steps": 1,
                "tokenizer_model_path": str(tmp_path / "tok" / "sp.model"),
                "dataset_config": {
                    "train_sources": [{"name": source.name, "index_path": source.index_path}],
                    "eval_sources": [],
                },
            },
        },
    }))
    monkeypatch.setattr(sys, "argv", ["tokenizer-training", "--config_file", str(config_file)])

    tokenizer_training.main()

    sp = sentencepiece.SentencePieceProcessor(model_file=str(tmp_path / "tok" / "sp.model"))
    assert sp.get_piece_size() == 40
