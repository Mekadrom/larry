import argparse
import csv
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from larry.common.config.config import ConfigValues
from larry.common.config.data.data_configs import DatasetMixtureConfig
from larry.common.config.training.definition import TrainingDefinition
from larry.common.data.dataloading.indexed_dataset import indexed_column, load_index, load_source_for_columns
from larry.common.utils.git_utils import WorkingTree
from larry.common.utils.init_helpers import import_submodules
from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.config.training.voice_trainer_configs import SpeechEncoderDecoderTrainerConfig

log = logging.getLogger(__name__)


def default_speaker_id(source: SpeechSourceConfig) -> str:
    return f"{source.name}_speaker_id"


class SpeakerMapping:
    """Wrapper/helper for a mapping of string "source:speaker_id" to unique, globally densified integer id."""

    def __init__(self) -> None:
        """Simple init."""
        self.ids: dict[str, int] = {}
        self.items: list[str] = []

    def add_item(self, item: str) -> int:
        """Returns the item's id, assigning the next one if it is new."""
        id_ = self.ids.get(item)
        if id_ is None:
            id_ = self.ids[item] = len(self.items)
            self.items.append(item)
        return id_

    def __getitem__(self, item: str) -> int:
        """Returns the integer id for an item."""
        return self.ids[item]

    def __len__(self) -> int:
        """Returns the number of unique items in this mapping."""
        return len(self.items)

    @staticmethod
    def namespace(source: SpeechSourceConfig) -> str:
        """The scope a source's speaker ids are unique within."""
        return source.speaker_namespace or source.name

    @staticmethod
    def key(source: SpeechSourceConfig, speaker_id: object) -> str:
        """The mapping key for a speaker id read from a source; single speaker sources ignore ``speaker_id``."""
        if source.speaker_id_column is None:
            speaker_id = default_speaker_id(source)
        return f"{SpeakerMapping.namespace(source)}:{speaker_id}"

    def save_csv(self, output_prefix: Path) -> None:
        """Generates a CSV file from this mapping."""
        with open(output_prefix.with_suffix(".csv"), mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["key", "id"])
            writer.writeheader()
            writer.writerows([{"key": item, "id": i} for item, i in self.ids.items()])

    @classmethod
    def from_csv(cls, file_path: Path) -> Self:
        """Creates a mapping from a saved CSV file."""
        instance = cls()
        with open(file_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = sorted((int(row["id"]), row["key"]) for row in reader)
            if [id_ for id_, _ in rows] != list(range(len(rows))):
                raise ValueError(f"{file_path} ids are not a dense 0..n-1 range")

            for _, key in rows:
                instance.add_item(key)

        return instance


class SpeakerMappingCreator:
    """Runner for producing a global speaker mapping from a specified training dataset."""

    def __init__(self, sources: Sequence[SpeechSourceConfig], output_prefix: Path) -> None:
        """Simple init."""
        self.sources = sources
        self.output_prefix = output_prefix

    def run(self) -> None:
        """Performs global speaker mapping by collecting sets of speaker IDs from dataset sources."""
        os.makedirs(self.output_prefix.parent, exist_ok=True)

        # sources sharing a namespace (e.g. a train split and its eval split) share speaker ids
        mapping = SpeakerMapping()
        for source in self.sources:
            for speaker_id in self.read_source(source):
                mapping.add_item(SpeakerMapping.key(source, speaker_id))

        mapping.save_csv(self.output_prefix)

    @staticmethod
    def read_source(source: SpeechSourceConfig) -> list[str] | list[None]:
        """For all examples in a provided source dataset, collect unique speaker_id occurrences."""
        if source.speaker_id_column is None:
            return [None]

        index = load_index(source)
        data = load_source_for_columns(source, index, [source.speaker_id_column])
        column = indexed_column(index, data, source.speaker_id_column).unique()
        # some sources store integer ids; a falsiness check would drop speaker 0
        return sorted({str(t) for t in column.to_pylist() if t is not None and t != ""})


def main() -> None:
    """Main entrypoint for building a speaker id map over an entire dataset mixture."""
    import_submodules("larry", skip=(
        "larry.common.training",
        "larry.voice.training",
        "larry.text.training",
        "larry.image.training"
    ))

    argparser = argparse.ArgumentParser(
        prog="speaker-mapping",
        description="Produces a global speaker mapping of (source, speaker_id_str): speaker_id_int.",
    )
    ConfigValues.add_arguments(argparser)
    argparser.add_argument(
        "--output_prefix",
        type=Path,
        required=False,
        help="Where to write <prefix>.csv; defaults to the config's speaker_mapping_path without its .csv suffix.",
    )
    args = argparser.parse_args()

    WorkingTree.ensure_clean(args.allow_dirty)
    definition = ConfigValues.from_args(args).load(TrainingDefinition)
    trainer_config = definition.trainer

    if not isinstance(trainer_config, SpeechEncoderDecoderTrainerConfig):
        raise TypeError(f"expected a SpeechEncoderDecoderTrainerConfig, got {type(trainer_config).__name__}")

    mixture_config: DatasetMixtureConfig[Any] = trainer_config.mixture_config

    output_prefix = args.output_prefix
    if output_prefix is None:
        if not trainer_config.speaker_mapping_path:
            raise ValueError("pass --output_prefix or set speaker_mapping_path in the training config")

        output_prefix = Path(trainer_config.speaker_mapping_path).with_suffix("")

    output_prefix = Path(output_prefix).expanduser().resolve()

    SpeakerMappingCreator(
        mixture_config.train_sources + mixture_config.eval_sources,
        output_prefix
    ).run()
