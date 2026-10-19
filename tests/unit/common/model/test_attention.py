import pytest
import torch

from larry.common.config.model.model_configs import AttentionModuleConfig
from larry.common.model.attention import LarryAttention


def _attention(n_query_groups: int, is_causal: bool) -> LarryAttention:
    torch.manual_seed(0)
    config = AttentionModuleConfig(
        d_model=64,
        n_heads=8,
        n_query_groups=n_query_groups,
        rotary_embedding_dim=8,
        attention_dropout_p=0.0,
        proj_dropout_p=0.0,
        is_causal=is_causal,
    )
    return LarryAttention(config).eval()


@pytest.mark.parametrize("n_query_groups", [1, 2, 8])
def test_value_heads_keep_the_full_head_width_for_any_group_count(n_query_groups: int) -> None:
    attention = _attention(n_query_groups, is_causal=False)

    assert attention.k_proj.out_features == n_query_groups * 8
    assert attention.v_proj.out_features == n_query_groups * 8
    assert attention.o_proj.in_features == 64


@pytest.mark.parametrize("n_query_groups", [1, 2, 8])
def test_padded_bidirectional_attention_runs_for_any_group_count(n_query_groups: int) -> None:
    attention = _attention(n_query_groups, is_causal=False)
    x = torch.randn(2, 5, 64)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    out, _ = attention(x, attention_mask=mask)

    assert out.shape == (2, 5, 64)
    assert torch.isfinite(out).all()


def test_grouped_attention_matches_full_attention_with_each_key_value_head_repeated() -> None:
    grouped = _attention(2, is_causal=False)
    full = _attention(8, is_causal=False)
    with torch.no_grad():
        full.q_proj.weight.copy_(grouped.q_proj.weight)
        full.o_proj.load_state_dict(grouped.o_proj.state_dict())
        for name in ("k_proj", "v_proj"):
            heads = getattr(grouped, name).weight.view(2, 8, 64)
            getattr(full, name).weight.copy_(heads.repeat_interleave(4, dim=0).reshape(64, 64))
    x = torch.randn(2, 5, 64)

    assert torch.allclose(grouped(x)[0], full(x)[0], atol=1e-5)


@pytest.mark.parametrize("n_query_groups", [1, 2, 8])
def test_cached_decoding_matches_a_full_causal_pass(n_query_groups: int) -> None:
    attention = _attention(n_query_groups, is_causal=True)
    x = torch.randn(1, 6, 64)

    full, _ = attention(x)
    steps = []
    cache = None
    for t in range(6):
        out, cache = attention(x[:, t:t + 1], kv_cache=cache, position_offset=t, use_cache=True)
        steps.append(out)

    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_cached_decode_of_a_left_padded_row_matches_the_full_forward() -> None:
    attention = _attention(n_query_groups=2, is_causal=True)
    x = torch.randn(2, 5, 64)
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])

    full, _ = attention(x, attention_mask=mask)
    _, cache = attention(x[:, :4], attention_mask=mask[:, :4], use_cache=True)
    last, _ = attention(x[:, 4:], attention_mask=mask, kv_cache=cache, position_offset=4)

    torch.testing.assert_close(last[:, 0], full[:, 4])


def test_the_cache_holds_one_head_per_query_group() -> None:
    attention = _attention(n_query_groups=2, is_causal=True)

    _, cache = attention(torch.randn(2, 4, 64), use_cache=True)

    assert cache is not None and cache.key_cache is not None and cache.value_cache is not None
    assert cache.key_cache.shape == (2, 2, 4, 8)
    assert cache.value_cache.shape == (2, 2, 4, 8)


def test_split_rope_cached_decode_offsets_the_global_positions() -> None:
    torch.manual_seed(0)
    config = AttentionModuleConfig(
        d_model=64,
        n_heads=8,
        n_query_groups=2,
        rotary_embedding_dim=8,
        use_split_rope=True,
        attention_dropout_p=0.0,
        proj_dropout_p=0.0,
        is_causal=True,
    )
    attention = LarryAttention(config).eval()
    x = torch.randn(2, 5, 64)
    global_positions = torch.arange(5.0).expand(2, 5)
    position_ids = torch.stack([global_positions, global_positions * 0.5], dim=-1)

    full, _ = attention(x, position_ids=position_ids)
    _, cache = attention(x[:, :4], position_ids=position_ids[:, :4], use_cache=True)
    decode_ids = position_ids[:, 4:].clone()
    decode_ids[..., 0] -= 4
    last, _ = attention(x[:, 4:], kv_cache=cache, position_offset=4, position_ids=decode_ids)

    torch.testing.assert_close(last[:, 0], full[:, 4])


def test_split_rope_without_position_ids_raises() -> None:
    config = AttentionModuleConfig(d_model=64, n_heads=8, n_query_groups=2, rotary_embedding_dim=8, use_split_rope=True)

    with pytest.raises(ValueError):
        LarryAttention(config)(torch.randn(1, 3, 64))


@pytest.mark.parametrize("is_causal", [False, True])
def test_the_qk_probe_scores_under_the_same_mask_sdpa_receives(
        monkeypatch: pytest.MonkeyPatch, is_causal: bool
) -> None:
    attention = _attention(n_query_groups=2, is_causal=is_causal)
    attention.qk_probe_active = True
    masks: dict[str, torch.Tensor | None] = {}
    masked_attention = attention._masked_attention
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def probe(queries: torch.Tensor, keys: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        masks["probe"] = attn_mask
        return masked_attention(queries, keys, attn_mask)

    def recording_sdpa(*args: object, attn_mask: torch.Tensor | None = None, **kwargs: object) -> torch.Tensor:
        masks["sdpa"] = attn_mask
        return sdpa(*args, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(attention, "_masked_attention", probe)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)

    attention(torch.randn(2, 5, 64), attention_mask=torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]))

    assert masks["probe"] is not None and masks["sdpa"] is not None
    assert torch.equal(masks["probe"], masks["sdpa"])
    assert attention.qk_max_logit is not None and torch.isfinite(attention.qk_max_logit).all()
