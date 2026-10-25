import dataclasses
from typing import Literal

import torch


@dataclasses.dataclass
class KVCache:
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        if self.key_cache is None:
            return 0

        return self.key_cache.shape[2]

    def update(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is None or self.value_cache is None:
            self.key_cache = new_keys
            self.value_cache = new_values
        else:
            self.key_cache = torch.cat([self.key_cache, new_keys], dim=2)
            self.value_cache = torch.cat([self.value_cache, new_values], dim=2)

        return self.key_cache, self.value_cache

    def clear(self) -> None:
        self.key_cache = None
        self.value_cache = None

    def extend_mask(self, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None or self.seq_len == 0:
            return mask

        ones = torch.ones(*mask.shape[:-1], self.seq_len, device=mask.device, dtype=mask.dtype)
        return torch.cat([ones, mask], dim=-1)

    def prefix(self, length: int) -> KVCache:
        if self.key_cache is None or self.value_cache is None:
            return KVCache()

        return KVCache(key_cache=self.key_cache[:, :, :length], value_cache=self.value_cache[:, :, :length])


@dataclasses.dataclass
class LayerKVCache:
    layer_caches: dict[int, KVCache] = dataclasses.field(default_factory=dict)

    def get_layer(self, layer_idx: int) -> KVCache:
        if layer_idx not in self.layer_caches:
            self.layer_caches[layer_idx] = KVCache()

        return self.layer_caches[layer_idx]

    @property
    def seq_len(self) -> int:
        if not self.layer_caches:
            return 0

        return next(iter(self.layer_caches.values())).seq_len

    def clear(self) -> None:
        for cache in self.layer_caches.values():
            cache.clear()

        self.layer_caches.clear()


@dataclasses.dataclass
class RecurrentKVCache:
    strategy: Literal["huginn", "per_iteration", "shared"] = "huginn"
    cache_budget: int = 16
    max_iterations: int = 64

    slots: dict[tuple[int, int], KVCache] = dataclasses.field(default_factory=dict, init=False)
    latest: dict[int, KVCache] = dataclasses.field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.strategy not in ("huginn", "per_iteration", "shared"):
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'huginn', 'per_iteration' or 'shared'.")

    def slot(self, iteration: int) -> int:
        if self.strategy == "huginn":
            return iteration % self.cache_budget

        if self.strategy == "per_iteration":
            return min(iteration, self.max_iterations - 1)

        return 0

    def read(self, iteration: int, layer_idx: int, length: int) -> KVCache:
        cached = self.slots.get((self.slot(iteration), layer_idx), KVCache()).prefix(length)
        if cached.seq_len == length:
            return cached

        fallback = self.latest.get(layer_idx, KVCache())
        if fallback.seq_len < length:
            raise ValueError(
                f"layer {layer_idx} holds {fallback.seq_len} cached positions but {length} were requested; "
                f"position_offset must equal the number of positions already processed"
            )

        missing = fallback.prefix(length)
        if cached.key_cache is None or cached.value_cache is None:
            return missing

        assert missing.key_cache is not None and missing.value_cache is not None
        return KVCache(
            key_cache=torch.cat([cached.key_cache, missing.key_cache[:, :, cached.seq_len:]], dim=2),
            value_cache=torch.cat([cached.value_cache, missing.value_cache[:, :, cached.seq_len:]], dim=2),
        )

    def write(self, iteration: int, layer_idx: int, cache: KVCache) -> None:
        self.slots[(self.slot(iteration), layer_idx)] = cache
        self.latest[layer_idx] = cache

    def copy(self) -> RecurrentKVCache:
        copied = RecurrentKVCache(
            strategy=self.strategy, cache_budget=self.cache_budget, max_iterations=self.max_iterations
        )
        # writes replace entries rather than mutating cached tensors, so a shallow copy is a snapshot
        copied.slots = dict(self.slots)
        copied.latest = dict(self.latest)
        return copied

    @property
    def seq_len(self) -> int:
        return max((cache.seq_len for cache in self.latest.values()), default=0)

    def clear(self) -> None:
        self.slots.clear()
        self.latest.clear()


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalModelKVCache:
    text_prelude_cache: list[KVCache | None] | None = None
    voice_prelude_cache: list[KVCache | None] | None = None
    voice_prelude_conv_state: torch.Tensor | None = None
    recurrent_cache: RecurrentKVCache | None = None
    text_coda_cache: list[KVCache | None] | None = None
    voice_coda_cache: list[KVCache | None] | None = None

    def copy(self) -> LarryMultimodalModelKVCache:
        recurrent_cache = self.recurrent_cache.copy() if self.recurrent_cache is not None else None
        return dataclasses.replace(self, recurrent_cache=recurrent_cache)
