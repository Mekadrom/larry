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
