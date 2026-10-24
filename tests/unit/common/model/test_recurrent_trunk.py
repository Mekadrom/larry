import pytest
import torch

from larry.common.config.model.model_configs import LarryRecurrentTrunkConfig
from larry.common.model import recurrent_criteria
from larry.common.model.recurrent import LarryRecurrentTrunk
from larry.common.model.recurrent_criteria import RecurrentExitCriteria, RecurrentExitCriteriaOutput

D_MODEL = 32


def _trunk(**overrides: object) -> LarryRecurrentTrunk:
    torch.manual_seed(0)
    return LarryRecurrentTrunk(LarryRecurrentTrunkConfig(**{
        "d_model": D_MODEL,
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
        "mean_thinking_steps": 6,
        "backprop_depth": 2,
        **overrides,
    }))


def _x(seq_len: int = 5) -> torch.Tensor:
    return torch.randn(2, seq_len, D_MODEL, generator=torch.Generator().manual_seed(1))


class ScriptedCriteria(RecurrentExitCriteria):
    def __init__(self, schedule: list[list[bool]]) -> None:
        super().__init__(0.0)
        self.schedule = schedule
        self.calls = 0

    def step(self, last_output: torch.Tensor, output: torch.Tensor) -> RecurrentExitCriteriaOutput:
        row = self.schedule[min(self.calls, len(self.schedule) - 1)]
        self.calls += 1
        converged = torch.tensor(row, dtype=torch.bool).expand(output.shape[0], -1)
        return RecurrentExitCriteriaOutput(converged_mask=converged, exit_values=torch.zeros(converged.shape))


def test_training_step_counts_are_reproducible_and_bounded_by_the_backprop_depth() -> None:
    first = _trunk(mean_thinking_steps=8, backprop_depth=3).train()
    second = _trunk(mean_thinking_steps=8, backprop_depth=3).train()

    draws = [first._n_k_steps(8, 3) for _ in range(200)]

    assert draws[:20] == [second._n_k_steps(8, 3) for _ in range(20)]
    assert first.step == 200
    assert all(0 < int(k) <= 3 and int(n) >= 0 for n, k in draws)
    assert all(int(n) == 0 or int(k) == 3 for n, k in draws)
    assert sum(int(n) + int(k) for n, k in draws) / len(draws) == pytest.approx(9.0, abs=1.0)


def test_evaluation_runs_the_mean_steps_without_backprop_and_does_not_advance_the_sampler() -> None:
    trunk = _trunk(mean_thinking_steps=7).eval()

    n, k = trunk._n_k_steps(7, 2)

    assert (int(n), int(k)) == (7, 0)
    assert trunk.step == 0


def test_a_training_forward_runs_the_sampled_steps_and_backpropagates() -> None:
    trunk = _trunk().train()
    expected_n, expected_k = _trunk().train()._n_k_steps(6, 2)

    out = trunk(_x())
    out.logits.square().mean().backward()

    assert out.n_iterations == int(expected_n) + int(expected_k)
    assert all(p.grad is not None for p in trunk.parameters() if p.requires_grad)


def test_gradient_checkpointing_gives_the_same_loss_and_gradients() -> None:
    plain = _trunk().train()
    checkpointed = _trunk(gradient_checkpointing=True).train()

    plain_loss = plain(_x()).logits.square().mean()
    checkpointed_loss = checkpointed(_x()).logits.square().mean()
    plain_loss.backward()
    checkpointed_loss.backward()

    assert torch.allclose(plain_loss, checkpointed_loss)
    for (name, a), (_, b) in zip(plain.named_parameters(), checkpointed.named_parameters(), strict=True):
        if a.requires_grad:
            assert torch.allclose(a.grad, b.grad, atol=1e-6), name


@pytest.mark.parametrize("override", [3, 9])
def test_evaluation_respects_the_iteration_override(override: int) -> None:
    trunk = _trunk(exit_criteria="none").eval()

    with torch.no_grad():
        assert trunk(_x(), max_iterations_override=override).n_iterations == override


def test_thought_state_initializations() -> None:
    x = _x() * 3
    states = {
        mode: _trunk(thought_state_init=mode).initialize_thinking_state(x)
        for mode in ("none", "normal", "embed", "like-init", "zero", "unit")
    }

    assert torch.equal(states["none"], x)
    assert not states["zero"].any()
    assert torch.allclose(states["unit"].mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(states["unit"].std(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert float(states["like-init"].abs().max()) <= 3 * 0.02 + 1e-6
    assert float(states["embed"].std()) == pytest.approx(1 / D_MODEL ** 0.5, rel=0.3)
    assert all(state.shape == x.shape for state in states.values())


def test_sum_injection_adds_the_input_to_the_thought_state() -> None:
    trunk = _trunk()
    x, thought = _x(), torch.randn(2, 5, D_MODEL)

    assert trunk.iteration_projection is None
    assert torch.equal(trunk._inject(x, thought), x + thought)


def test_concat_injection_starts_out_equal_to_sum() -> None:
    concat = _trunk(thought_state_injection_type="concat")
    x, thought = _x(), torch.randn(2, 5, D_MODEL)

    assert tuple(concat.iteration_projection.weight.shape) == (D_MODEL, 2 * D_MODEL)
    assert torch.allclose(concat._inject(x, thought), x + thought, atol=1e-6)


@pytest.mark.parametrize(("placement", "norm_width"), [(None, None), ("pre", 2 * D_MODEL), ("post", D_MODEL)])
def test_concat_projection_norm_placement(placement: str | None, norm_width: int | None) -> None:
    trunk = _trunk(thought_state_injection_type="concat", iteration_projection_norm_placement=placement)
    x, thought = _x() * 5, torch.randn(2, 5, D_MODEL) * 5

    injected = trunk._inject(x, thought)

    norms = [n for n in (trunk.pre_iteration_projection_norm, trunk.post_iteration_projection_norm) if n is not None]
    assert [n.normalized_shape[0] for n in norms] == ([] if norm_width is None else [norm_width])
    if placement == "post":
        assert torch.allclose(injected.square().mean(dim=-1), torch.ones(2, 5), atol=1e-3)
    if placement is None:
        assert torch.allclose(injected, x + thought, atol=1e-5)


def test_ada_norm_is_rejected_for_the_projection() -> None:
    with pytest.raises(ValueError, match="doesn't support"):
        _trunk(
            thought_state_injection_type="concat",
            iteration_projection_norm_placement="pre",
            iteration_projection_norm_type="ada",
        )


@pytest.mark.parametrize(("criteria", "expected"), [
    ("latent-diff", recurrent_criteria.LatentDiffCriteria),
    ("logit-kl", recurrent_criteria.LogitKLCriteria),
    ("none", recurrent_criteria.NoOpCriteria),
])
def test_exit_criteria_follow_the_config(criteria: str, expected: type) -> None:
    trunk = _trunk(
        exit_criteria=criteria, latent_diff_exit_criteria_threshold=0.2, logits_kl_exit_criteria_threshold=0.1
    )

    assert type(trunk.exit_criteria) is expected
    if criteria != "none":
        assert trunk.exit_criteria.threshold in (0.2, 0.1)


def test_no_exit_criteria_runs_every_iteration() -> None:
    trunk = _trunk(exit_criteria="none").eval()

    with torch.no_grad():
        assert trunk(_x()).n_iterations == 6


def test_logit_kl_requires_a_readout_outside_training() -> None:
    trunk = _trunk(exit_criteria="logit-kl").eval()

    with pytest.raises(ValueError, match="readout"):
        trunk(_x())


def test_logit_kl_does_not_need_a_readout_in_training() -> None:
    assert _trunk(exit_criteria="logit-kl").train()(_x()).n_iterations > 0


def test_logit_kl_stops_once_the_readout_stops_changing() -> None:
    trunk = _trunk(exit_criteria="logit-kl").eval()
    fixed = torch.linspace(-3, 3, 10)

    with torch.no_grad():
        out = trunk(_x(), readout=lambda h: fixed.expand(*h.shape[:-1], 10))

    assert out.n_iterations == 2


def test_all_tokens_converging_stops_early_and_keeps_their_states() -> None:
    trunk = _trunk().eval()
    trunk.exit_criteria = ScriptedCriteria([[False] * 5, [True] * 5])
    reference = _trunk(exit_criteria="none").eval()

    with torch.no_grad():
        out = trunk(_x())
        two_steps = reference(_x(), max_iterations_override=2).logits

    assert out.n_iterations == 2
    assert torch.allclose(out.logits, two_steps, atol=1e-6)


def test_converged_tokens_are_frozen_while_the_rest_keep_iterating() -> None:
    trunk = _trunk().eval()
    trunk.exit_criteria = ScriptedCriteria([[True, False, False, False, False], [False] * 5])
    reference = _trunk(exit_criteria="none").eval()

    with torch.no_grad():
        out = trunk(_x())
        one_step = reference(_x(), max_iterations_override=1).logits
        full = reference(_x()).logits

    assert out.n_iterations == 6
    assert torch.allclose(out.logits[:, 0], one_step[:, 0], atol=1e-6)
    assert not torch.allclose(out.logits[:, 0], full[:, 0])
    assert not torch.allclose(out.logits[:, 1:], full[:, 1:])


def test_tokens_that_are_not_eligible_never_converge() -> None:
    trunk = _trunk().eval()
    trunk.exit_criteria = ScriptedCriteria([[True] * 5])
    eligible = torch.tensor([[True, True, False, False, False]] * 2)

    reference = _trunk(exit_criteria="none").eval()

    with torch.no_grad():
        out = trunk(_x(), converge_eligible=eligible)
        one_step = reference(_x(), max_iterations_override=1).logits

    assert out.n_iterations == 6
    assert torch.allclose(out.logits[:, :2], one_step[:, :2], atol=1e-6)
    assert not torch.allclose(out.logits[:, 2:], one_step[:, 2:])


def test_latent_diff_measures_the_relative_change() -> None:
    criteria = recurrent_criteria.LatentDiffCriteria(0.1)
    previous = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    current = torch.tensor([[[1.05, 0.0], [2.0, 0.0]]])

    out = criteria.step(previous, current)

    assert out.exit_values[0].tolist() == pytest.approx([0.05 / 1.05, 0.5])
    assert out.converged_mask[0].tolist() == [True, False]


def test_no_op_criteria_never_converges() -> None:
    out = recurrent_criteria.NoOpCriteria().step(torch.zeros(1, 3, 4), torch.zeros(1, 3, 4))

    assert not out.converged_mask.any()


def test_logit_kl_is_zero_for_identical_distributions_and_starts_from_uniform() -> None:
    criteria = recurrent_criteria.LogitKLCriteria(1e-3)
    log_probs = torch.log_softmax(torch.tensor([[[2.0, 0.0, -1.0]]]), dim=-1)

    assert criteria.step(log_probs, log_probs).converged_mask.all()
    assert not criteria.step(criteria.init_state(log_probs), log_probs).converged_mask.any()
