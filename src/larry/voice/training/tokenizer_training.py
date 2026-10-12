import argparse
import collections
import dataclasses
import glob
import logging
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import sentencepiece

from larry.common.config.training import build_model_config, build_trainer_config
from larry.common.utils import load_yaml_config
from larry.common.utils.init_helpers import import_submodules
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.config.training.voice_trainer_configs import SpeechTokenizerTrainerConfig
from larry.voice.data.dataloading.voice_dataloading import SpeechSourceConfig

log = logging.getLogger(__name__)


@dataclasses.dataclass
class SourceText:
    name: str
    weight: float
    seconds: float
    counts: collections.Counter[str]


def split_files(index_path: str, split: str) -> list[str]:
    files = sorted(glob.glob(str(Path(index_path).expanduser() / split / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files for split {split!r} under {index_path}")
    return files


def read_source(source: SpeechSourceConfig) -> SourceText:
    counts: collections.Counter[str] = collections.Counter()
    seconds = 0.0
    columns = [source.text_column, source.duration_column]
    for file in split_files(source.index_path, source.split):
        for batch in pq.ParquetFile(file).iter_batches(columns=columns, batch_size=65536):
            seconds += pc.sum(batch.column(source.duration_column)).as_py() or 0.0
            counts.update(t for t in batch.column(source.text_column).to_pylist() if t)
    return SourceText(source.name, source.weight, seconds, counts)


def weighted_frequencies(sources: list[SourceText]) -> collections.Counter[str]:
    total_weight = sum(s.weight for s in sources if s.seconds > 0)
    total_seconds = sum(s.seconds for s in sources)
    factors = {
        s.name: (s.weight / total_weight) / (s.seconds / total_seconds)
        for s in sources
        if s.seconds > 0 and s.weight > 0
    }
    if not factors:
        raise ValueError("no source has both audio and a positive weight")
    scale = 1.0 / min(factors.values())

    frequencies: collections.Counter[str] = collections.Counter()
    for s in sources:
        if s.name not in factors:
            continue
        per_occurrence = factors[s.name] * scale
        for text, n in s.counts.items():
            frequencies[text] += round(n * per_occurrence)
    return collections.Counter({t: max(1, round(f)) for t, f in frequencies.items()})


def write_tsv(frequencies: collections.Counter[str], tsv: Path) -> None:
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv, "w", encoding="utf-8") as f:
        for text, frequency in frequencies.items():
            if "\t" in text or "\n" in text:
                raise ValueError(f"text contains a tab or newline and cannot be written as TSV: {text!r}")
            f.write(f"{text}\t{frequency}\n")


def train_tokenizer(sources: list[SpeechSourceConfig], output_prefix: Path, vocab_size: int) -> Path:
    texts = [read_source(s) for s in sources]
    for t in texts:
        log.info(f"{t.name}: {t.seconds / 3600:.1f} h, {len(t.counts)} distinct lines, weight={t.weight}")

    tsv = output_prefix.with_suffix(".tsv")
    write_tsv(weighted_frequencies(texts), tsv)

    sentencepiece.SentencePieceTrainer.train(
        input=str(tsv),
        input_format="tsv",
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    return output_prefix.with_suffix(".model")


def main() -> None:
    import_submodules("larry", skip=("larry.common.training", "larry.voice.training", "larry.text.training",
                                     "larry.image.training"))

    argparser = argparse.ArgumentParser(
        prog="tokenizer-training",
        description="Trains the SentencePiece text tokenizer for a speech tokenizer training config, weighting each "
                    "train source's transcripts by its mixture weight.",
    )
    argparser.add_argument(
        "--config_file",
        type=Path,
        required=True,
        help="The training config.yaml whose train_sources and model text_vocab_size to use.",
    )
    argparser.add_argument(
        "--output_prefix",
        type=Path,
        required=False,
        help="Where to write <prefix>.model, <prefix>.vocab, and <prefix>.tsv; defaults to the config's "
             "tokenizer_model_path without its .model suffix.",
    )
    args = argparser.parse_args()

    definition = load_yaml_config(Path(args.config_file).expanduser().resolve())
    trainer_config = build_trainer_config(definition.get("trainer", {}))
    model_config = build_model_config(definition.get("model", {}))
    if not isinstance(trainer_config, SpeechTokenizerTrainerConfig):
        raise TypeError(f"expected a SpeechTokenizerTrainerConfig, got {type(trainer_config).__name__}")
    if not isinstance(model_config, SpeechTokenizerModelConfig):
        raise TypeError(f"expected a SpeechTokenizerModelConfig, got {type(model_config).__name__}")

    output_prefix = args.output_prefix
    if output_prefix is None:
        if not trainer_config.tokenizer_model_path:
            raise ValueError("pass --output_prefix or set tokenizer_model_path in the training config")
        output_prefix = Path(trainer_config.tokenizer_model_path).with_suffix("")
    output_prefix = Path(output_prefix).expanduser().resolve()

    model_file = train_tokenizer(trainer_config.mixture_config.train_sources, output_prefix, model_config.text_vocab_size)
    log.info(f"wrote {model_file}")
