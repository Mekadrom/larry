import itertools
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from accelerate.state import AcceleratorState
from accelerate.utils import PrepareForLaunch, patch_environment
from torch.multiprocessing import start_processes

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


def spawn_launcher(function: Callable[..., None], args: tuple[Any, ...] = (), num_processes: int = 2) -> None:
    # accelerate's debug_launcher forks, which breaks once this pytest process has run autograd; spawned workers
    # start clean, at the cost of re-importing, so function must be importable at module level
    with tempfile.NamedTemporaryFile() as rendezvous, patch_environment(
        world_size=num_processes,
        master_addr="127.0.0.1",
        master_port="29500",
        accelerate_mixed_precision="no",
        accelerate_debug_rdv_file=rendezvous.name,
        accelerate_use_cpu="yes",
        gloo_socket_ifname="lo",
    ):
        start_processes(PrepareForLaunch(function, debug=True), args=args, nprocs=num_processes, start_method="spawn")
