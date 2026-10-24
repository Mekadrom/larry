import argparse
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import sentencepiece
from datasets import Dataset as HFDataset

from larry.common.config.config import ConfigValues
from larry.common.config.data.data_configs import DatasetMixtureConfig, TextSourceConfig
from larry.common.config.model.model_configs import VocabularyConfig
from larry.common.config.training.definition import TrainingDefinition
from larry.common.config.training.trainer_configs import TextTokenizerConfig
from larry.common.data.dataloading.indexed_dataset import indexed_column, load_index, load_source_for_columns
from larry.common.utils.git_utils import WorkingTree
from larry.common.utils.init_helpers import import_submodules
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry

log = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class SourceText:
    """Helper dataclass for counting how many times the exact same sentence has occurred."""

    name: str
    weight: float
    counts: Counter[str]
    size: float
    unit: str


class SourceTextReader[S: TextSourceConfig = TextSourceConfig](
    Registrable, ABC, root=True, config_root=TextSourceConfig
):
    REGISTRY: ClassVar[TypeRegistry[SourceTextReader]]

    unit: ClassVar[str]

    def columns(self, source: S) -> list[str]:
        return [source.text_column]

    def read(self, source: S) -> SourceText:
        index = load_index(source)
        data = load_source_for_columns(source, index, self.columns(source))
        texts = [t for t in indexed_column(index, data, source.text_column).to_pylist() if t]
        return SourceText(
            name=source.name,
            weight=source.weight,
            counts=Counter(texts),
            size=self.size(source, index, data, texts),
            unit=self.unit,
        )

    @abstractmethod
    def size(self, source: S, index: HFDataset, data: HFDataset | None, texts: list[str]) -> float:
        ...


class TokenizerTrainer:
    def __init__(self, sources: Sequence[TextSourceConfig], output_prefix: Path, vocab_size: int) -> None:
        self.sources = sources
        self.output_prefix = output_prefix
        self.vocab_size = vocab_size

    def run(self) -> Path:
        """Trains a sentence piece tokenizer by delegating to ``sentencePiece.SentencePieceTrainer``."""
        texts = [self.read_source(s) for s in self.sources]
        for t in texts:
            log.info(f"{t.name}: {t.size:.1f} {t.unit}, {len(t.counts)} distinct lines, weight={t.weight}")

        tsv = self.output_prefix.with_suffix(".tsv")
        self._write_tsv(self._weighted_frequencies(texts), tsv)

        sentencepiece.SentencePieceTrainer.train(
            input=str(tsv),
            input_format="tsv",
            model_prefix=str(self.output_prefix),
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            train_extremely_large_corpus=True,
        )
        return self.output_prefix.with_suffix(".model")

    @staticmethod
    def read_source(source: TextSourceConfig) -> SourceText:
        return SourceTextReader.REGISTRY.for_config(source)().read(source)

    @staticmethod
    def _write_tsv(frequencies: Counter[str], tsv: Path) -> None:
        """Exports the frequencies to a TSV file that the SentencePiece trainer can ingest."""
        tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(tsv, "w", encoding="utf-8") as f:
            for text, frequency in frequencies.items():
                if "\t" in text or "\n" in text:
                    raise ValueError(f"text contains a tab or newline and cannot be written as TSV: {text!r}")
                f.write(f"{text}\t{frequency}\n")

    @staticmethod
    def _factors(sources: list[SourceText]) -> dict[str, float]:
        weighted = [s for s in sources if s.weight > 0 and s.size > 0]
        units = {s.unit for s in weighted}
        if len(units) > 1:
            raise ValueError(f"mixture weights must share one unit to be comparable, got {sorted(units)}")

        total_weight = sum(s.weight for s in weighted)
        total_size = sum(s.size for s in weighted)
        return {
            s.name: (s.weight / total_weight) / (s.size / total_size)
            for s in weighted
        }

    def _weighted_frequencies(self, sources: list[SourceText]) -> Counter[str]:
        """Produced frequencies per sentence weighted by the size of the source dataset."""
        factors = self._factors(sources)
        if not factors:
            raise ValueError("no source has a positive weight")

        scale = 1.0 / min(factors.values())

        frequencies: Counter[str] = Counter()
        for s in sources:
            if s.name not in factors:
                continue

            per_occurrence = factors[s.name] * scale
            for text, n in s.counts.items():
                frequencies[text] += round(n * per_occurrence)

        return Counter({
            t: max(1, round(f))
            for t, f in frequencies.items()
        })


def main() -> None:
    """Main entrypoint for training a tokenizer on a target training config's exact weighted dataset mixture."""
    import_submodules("larry", skip=(
        "larry.common.training",
        "larry.voice.training",
        "larry.text.training",
        "larry.image.training"
    ))

    argparser = argparse.ArgumentParser(
        prog="tokenizer-training",
        description="Trains the SentencePiece text tokenizer for a training config, weighting each train source's "
                    "text by its mixture weight.",
    )
    ConfigValues.add_arguments(argparser)
    argparser.add_argument(
        "--output_prefix",
        type=Path,
        required=False,
        help="Where to write <prefix>.model, <prefix>.vocab, and <prefix>.tsv; defaults to the config's "
             "tokenizer_model_path without its .model suffix.",
    )
    args = argparser.parse_args()

    WorkingTree.ensure_clean(args.allow_dirty)
    definition = ConfigValues.from_args(args).load(TrainingDefinition)
    trainer_config = definition.trainer
    model_config = definition.model

    if not isinstance(model_config, VocabularyConfig):
        raise TypeError(f"{type(model_config).__name__} has no text vocabulary to train a tokenizer for")

    output_prefix = args.output_prefix
    if output_prefix is None:
        if not isinstance(trainer_config, TextTokenizerConfig):
            raise ValueError(f"pass --output_prefix; {type(trainer_config).__name__} has no tokenizer_model_path")

        output_prefix = Path(trainer_config.tokenizer_model_path).with_suffix("")

    mixture_config: DatasetMixtureConfig[Any] = trainer_config.mixture_config
    sources = mixture_config.train_sources
    textless = [s.name for s in sources if not isinstance(s, TextSourceConfig)]
    if textless:
        raise TypeError(f"train sources without a text column: {textless}")

    output_prefix = Path(output_prefix).expanduser().resolve()
    model_file = TokenizerTrainer(sources, output_prefix, model_config.vocab_size).run()
    log.info(f"wrote {model_file}")
