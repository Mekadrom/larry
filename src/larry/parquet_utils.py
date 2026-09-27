from pathlib import Path

from datasets import Dataset


def save_parquet_shards(ds: Dataset, parquet_dir: Path, parquet_target_size_mb: float, prefix: str) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, min(len(ds), ds.data.nbytes // (parquet_target_size_mb * 1024 ** 2)))
    for i in range(num_shards):
        shard = ds.shard(num_shards, i, contiguous=True)
        shard.to_parquet(
            parquet_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet"
        )
    (parquet_dir / f"{prefix}.parquets.complete").touch(exist_ok=True)
