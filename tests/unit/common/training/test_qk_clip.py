import torch
from accelerate import Accelerator
from torch import nn

from larry.common.config.model.model_configs import AttentionModuleConfig
from larry.common.model.attention import LarryAttention
from larry.common.training.qk_clip import QKClipHandler
from tests.helpers.common import spawn_launcher


def _attention_pair() -> nn.ModuleDict:
    torch.manual_seed(0)
    config = AttentionModuleConfig(d_model=8, n_heads=2, n_query_groups=2, rotary_embedding_dim=4)
    return nn.ModuleDict({"a": LarryAttention(config), "b": LarryAttention(config)})


def _armed(model: nn.ModuleDict, a: list[float] | None, b: list[float] | None) -> QKClipHandler:
    handler = QKClipHandler(model, tau=50.0, probe_every=1)
    handler.maybe_arm(0)
    model["a"].qk_max_logit = None if a is None else torch.tensor(a)
    model["b"].qk_max_logit = None if b is None else torch.tensor(b)
    return handler


def _head_norms(attention: LarryAttention) -> torch.Tensor:
    q = attention.q_proj.weight.view(2, 4, 8).norm(dim=(1, 2))
    k = attention.k_proj.weight.view(2, 4, 8).norm(dim=(1, 2))
    return q * k


def test_heads_over_tau_shrink_to_it_and_modules_that_did_not_run_are_left_alone() -> None:
    model = _attention_pair()
    before = {name: _head_norms(m) for name, m in model.items()}

    metrics = _armed(model, [100.0, 10.0], None).apply()

    assert torch.allclose(_head_norms(model["a"]) / before["a"], torch.tensor([0.5, 1.0]))
    assert torch.allclose(_head_norms(model["b"]), before["b"])
    assert metrics["qk_clip/heads_total"].value == 2
    assert metrics["qk_clip/max_logit"].value == 100.0


def _clip_with_a_different_module_on_each_rank() -> None:
    model = _attention_pair()
    before = {name: _head_norms(m) for name, m in model.items()}
    rank = Accelerator(cpu=True).process_index

    _armed(model, [100.0, 10.0] if rank == 0 else None, None if rank == 0 else [80.0, 5.0]).apply()

    assert torch.allclose(_head_norms(model["a"]) / before["a"], torch.tensor([0.5, 1.0])), rank
    assert torch.allclose(_head_norms(model["b"]) / before["b"], torch.tensor([50 / 80, 1.0])), rank


def test_ranks_that_ran_different_modules_reduce_together_and_clip_alike() -> None:
    spawn_launcher(_clip_with_a_different_module_on_each_rank)
