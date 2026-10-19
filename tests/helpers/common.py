import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from accelerate.state import AcceleratorState

from larry.common.config.data.preprocessing.pipeline_config import Provenance
from larry.common.utils.init_helpers import import_submodules


def import_larry_submodules() -> None:
    import_submodules("larry", skip=(
        "larry.common.training",
        "larry.voice.training",
        "larry.text.training",
        "larry.image.training",
    ))


def provenance() -> Provenance:
    return Provenance()


def reset_accelerate_state() -> None:
    AcceleratorState._reset_state(reset_partial_state=True)


def uninitialized[T](cls: type[T], **attributes: Any) -> T:
    instance = cls.__new__(cls)
    vars(instance).update(attributes)
    return instance


def first[T](iterable: Iterable[T], n: int) -> list[T]:
    return list(itertools.islice(iterable, n))


def write_parquet_split(root: Path, split: str, columns: dict[str, list[Any]], files: int = 1) -> Path:
    (root / split).mkdir(parents=True, exist_ok=True)
    rows = len(next(iter(columns.values())))
    per_file = -(-rows // files)
    for i in range(files):
        chunk = {name: values[i * per_file:(i + 1) * per_file] for name, values in columns.items()}
        pq.write_table(pa.table(chunk), root / split / f"{split}-{i:05d}-of-{files:05d}.parquet")
    return root


def write_parquet_splits(root: Path, splits: dict[str, dict[str, list[Any]]]) -> Path:
    for split, columns in splits.items():
        write_parquet_split(root, split, columns)
    return root
