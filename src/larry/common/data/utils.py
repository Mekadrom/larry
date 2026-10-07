from pathlib import Path
from typing import Mapping

import numpy as np
from datasets import Dataset, DatasetDict, Features
from tqdm import tqdm


def save_parquet_shards(ds: Dataset, parent_dir: Path, parquet_target_size_mb: float, prefix: str) -> None:
    parquet_dir = parent_dir / prefix
    parquet_dir.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, min(len(ds), ds.data.nbytes // (parquet_target_size_mb * 1024 ** 2)))
    for i in tqdm(range(num_shards)):
        shard = ds.shard(num_shards, i, contiguous=True)
        shard.to_parquet(
            parquet_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet"
        )
    (parent_dir / f"{prefix}.parquets.complete").touch(exist_ok=True)


def split_by_fractions(dataset: Dataset, batch_size: int, fractions: Mapping[str, float],
                       seed: int = 42) -> DatasetDict:
    dataset = dataset.with_format("arrow").shuffle(seed=seed).map(
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size
    )

    n = len(dataset)

    # eg for fractions like {"train": 0.98, "val": 0.01, "test": 0.01} produces an array of [0.0, 0.98, 0.99, 1.00]
    # and then multiplies to the length of the dataset to get something like [0, 9800, 9900, 10000] (for a dataset
    # of length 10,000) representing each bound of a ranged indices for each split requested (has to cast to int for
    # rounding)
    indices = np.cumsum([0.0, *fractions.values()]) * n
    indices = np.round(indices).astype(int)
    indices[-1] = n
    return DatasetDict({
        name: dataset.select(range(int(a), int(b)))
        for name, a, b in zip(fractions, indices[:-1], indices[1:])
    })


def features_of(dataset: Dataset | DatasetDict) -> Features:
    if isinstance(dataset, Dataset):
        return dataset.features
    schemas = {name: split.features for name, split in dataset.items()}

    # just check that the features all match in each split in the dataset dict, which allows us to return any of them
    first_name, first = next(iter(schemas.items()))
    for name, feats in schemas.items():
        if feats != first:
            raise ValueError(f"split {name!r} features differ from {first_name!r}: {feats} vs {first}")
    return first
