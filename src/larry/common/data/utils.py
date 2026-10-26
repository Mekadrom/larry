from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, Features
from tqdm import tqdm


def _write_shard(ds: Dataset, num_shards: int, i: int, parquet_dir: Path, prefix: str) -> None:
    """Helper for writing a dataset shard to file with a specific naming scheme."""
    shard = ds.shard(num_shards, i, contiguous=True)
    shard.to_parquet(
        parquet_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet"
    )


def save_parquet_shards(
        ds: Dataset,
        parent_dir: Path,
        parquet_target_size_mb: float,
        prefix: str,
        writer_batch_size: int,
        num_workers: int = 8,
) -> None:
    """Saves a dataset as manually sharded parquet files in the standard HF parquet format to disk.

    Parallelizes saving using a thread pool.
    """
    parquet_dir = parent_dir / prefix
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # noinspection PyProtectedMember
    if ds._indices is not None:
        ds = ds.flatten_indices(num_proc=num_workers, writer_batch_size=writer_batch_size)

    num_shards = int(max(1, min(len(ds), ds.data.nbytes // (parquet_target_size_mb * 1024 ** 2))))

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_write_shard, ds, num_shards, i, parquet_dir, prefix)
            for i in range(num_shards)
        ]
        for future in tqdm(as_completed(futures), total=num_shards):
            future.result()

    (parent_dir / f"{prefix}.parquets.complete").touch(exist_ok=True)


def features_of(dataset: Dataset | DatasetDict) -> Features:
    """Extracts features from a dataset, and ensures that each split in a ``DatasetDict`` have the same features."""
    if isinstance(dataset, Dataset):
        return dataset.features
    schemas = {name: split.features for name, split in dataset.items()}

    # just check that the features all match in each split in the dataset dict, which allows us to return any of them
    first_name, first = next(iter(schemas.items()))
    for name, feats in schemas.items():
        if feats != first:
            raise ValueError(f"split {name!r} features differ from {first_name!r}: {feats} vs {first}")
    return first


def bot_user_agent(name: str) -> str:
    return f"{name}-bot/1.0 (https://github.com/Mekadrom; dataset research) python-requests/{requests.__version__}"
