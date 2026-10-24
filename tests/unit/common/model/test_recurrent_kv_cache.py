import pytest
import torch

from larry.common.config.model.model_configs import LarryRecurrentTrunkConfig, LLMBlockModuleConfig
from larry.common.model.kv_cache import KVCache, RecurrentKVCache
from larry.common.model.llm import LarrySimpleDecoderBlock
from larry.common.model.recurrent import LarryRecurrentTrunk


def _kv(length: int, fill: float) -> KVCache:
    return KVCache(key_cache=torch.full((1, 2, length, 4), fill), value_cache=torch.full((1, 2, length, 4), -fill))


def test_prefix_slices_positions_and_handles_an_empty_cache() -> None:
    assert _kv(5, 1.0).prefix(3).seq_len == 3
    assert KVCache().prefix(3).seq_len == 0


@pytest.mark.parametrize(("strategy", "slots"), [
    ("huginn", [0, 1, 2, 3, 0, 1]),
    ("per_iteration", [0, 1, 2, 3, 4, 4]),
    ("shared", [0, 0, 0, 0, 0, 0]),
])
def test_iterations_map_to_slots_by_strategy(strategy: str, slots: list[int]) -> None:
    cache = RecurrentKVCache(strategy=strategy, cache_budget=4, max_iterations=5)

    assert [cache.slot(i) for i in range(6)] == slots


def test_an_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        RecurrentKVCache(strategy="bogus")


def test_a_read_returns_only_the_positions_before_the_offset() -> None:
    cache = RecurrentKVCache(cache_budget=2)
    cache.write(0, 0, _kv(5, 1.0))

    assert cache.read(0, 0, 3).seq_len == 3
    assert cache.read(2, 0, 5).seq_len == 5


def test_missing_positions_are_filled_from_the_latest_write() -> None:
    cache = RecurrentKVCache(strategy="per_iteration")
    cache.write(0, 0, _kv(3, 1.0))
    cache.write(1, 0, _kv(5, 2.0))

    filled = cache.read(0, 0, 5)

    assert filled.key_cache is not None
    assert filled.key_cache[0, 0, :, 0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    assert cache.read(4, 0, 5).key_cache[0, 0, :, 0].tolist() == [2.0] * 5


def test_reading_past_what_was_ever_cached_is_an_error() -> None:
    cache = RecurrentKVCache()
    cache.write(0, 0, _kv(3, 1.0))

    with pytest.raises(ValueError, match="position_offset"):
        cache.read(0, 0, 4)
    assert cache.read(0, 1, 0).seq_len == 0


def _trunk(**overrides: object) -> LarryRecurrentTrunk:
    torch.manual_seed(0)
    config = LarryRecurrentTrunkConfig(**{
        "d_model": 32,
        "n_heads": 4,
        "n_query_groups": 4,
        "rotary_embedding_dim": 8,
        "n_layers": 2,
        "max_seq_len": 64,
        "is_causal": True,
        "attention_dropout_p": 0.0,
        "proj_dropout_p": 0.0,
        "mlp_dropout_p": 0.0,
        "thought_state_init": "zero",
        "latent_diff_exit_criteria_threshold": 0.0,
        "mean_thinking_steps": 6,
        **overrides,
    })
    return LarryRecurrentTrunk(config).eval()


def _decode(trunk: LarryRecurrentTrunk, x: torch.Tensor, prefill: int, **kwargs: object) -> torch.Tensor:
    with torch.no_grad():
        first = trunk(x[:, :prefill], use_cache=True, **kwargs)
        cache = first.kv_cache
        outputs = [first.logits]
        for t in range(prefill, x.shape[1]):
            step = trunk(x[:, t:t + 1], kv_cache=cache, use_cache=True, position_offset=t, **kwargs)
            outputs.append(step.logits)
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("strategy", ["per_iteration", "huginn"])
def test_cached_decoding_matches_a_full_forward_when_every_iteration_has_its_own_slot(strategy: str) -> None:
    trunk = _trunk(kv_cache_strategy=strategy, kv_cache_budget=8)
    x = torch.randn(2, 7, 32)

    with torch.no_grad():
        full = trunk(x).logits

    assert torch.allclose(_decode(trunk, x, prefill=3), full, atol=1e-5)


def test_more_iterations_than_slots_never_duplicates_positions() -> None:
    trunk = _trunk(kv_cache_budget=4, mean_thinking_steps=10)
    x = torch.randn(2, 6, 32)

    with torch.no_grad():
        out = trunk(x, use_cache=True)
        step = trunk(torch.randn(2, 1, 32), kv_cache=out.kv_cache, use_cache=True, position_offset=6)

    cache = step.kv_cache
    assert cache is not None
    assert {key: kv.seq_len for key, kv in cache.slots.items()} == {(s, layer): 7 for s in range(4) for layer in range(2)}
    assert cache.seq_len == 7


def test_chunks_that_run_more_iterations_than_earlier_ones_read_filled_slots() -> None:
    trunk = _trunk(kv_cache_strategy="per_iteration")
    x = torch.randn(1, 5, 32)

    with torch.no_grad():
        prefill = trunk(x[:, :4], use_cache=True, max_iterations_override=2)
        step = trunk(x[:, 4:], kv_cache=prefill.kv_cache, use_cache=True, position_offset=4, max_iterations_override=5)

    cache = step.kv_cache
    assert cache is not None
    assert step.n_iterations == 5
    assert all(kv.seq_len == 5 for kv in cache.slots.values())
    assert sorted({slot for slot, _ in cache.slots}) == [0, 1, 2, 3, 4]


def test_the_shared_strategy_keeps_one_slot_per_layer() -> None:
    trunk = _trunk(kv_cache_strategy="shared")

    with torch.no_grad():
        cache = trunk(torch.randn(1, 4, 32), use_cache=True).kv_cache

    assert cache is not None
    assert sorted(cache.slots) == [(0, 0), (0, 1)]


def test_a_new_cache_takes_its_settings_from_the_config() -> None:
    trunk = _trunk(kv_cache_strategy="per_iteration", kv_cache_budget=3, kv_cache_max_iterations=9)

    with torch.no_grad():
        cache = trunk(torch.randn(1, 2, 32), use_cache=True).kv_cache

    assert cache is not None
    assert (cache.strategy, cache.cache_budget, cache.max_iterations) == ("per_iteration", 3, 9)


def test_training_never_builds_a_cache() -> None:
    trunk = _trunk().train()

    assert trunk(torch.randn(1, 3, 32), use_cache=True).kv_cache is None


def test_cross_attention_reuses_the_cached_encoder_keys() -> None:
    torch.manual_seed(0)
    block = LarrySimpleDecoderBlock(LLMBlockModuleConfig(
        d_model=32, n_heads=4, n_query_groups=4, rotary_embedding_dim=8, max_seq_len=32, is_causal=True,
        attention_dropout_p=0.0, proj_dropout_p=0.0, mlp_dropout_p=0.0,
    )).eval()
    x = torch.randn(1, 5, 32)
    encoder = torch.randn(1, 9, 32)

    with torch.no_grad():
        full = block(x, encoder).logits
        caches = None
        steps = []
        for t in range(5):
            out = block(x[:, t:t + 1], encoder, position_offset=t, kv_caches=caches, use_cache=True)
            caches = out.kv_cache
            steps.append(out.logits)

    assert caches is not None and caches[1] is not None
    assert caches[0].seq_len == 5
    assert caches[1].seq_len == 9
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)
