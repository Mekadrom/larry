import math

import pytest
import torch

from larry.common.config.model.model_configs import FlowMatchingConfig
from larry.common.model.flow import ConditionalFlowMatching


def _flow(**overrides: object) -> ConditionalFlowMatching:
    return ConditionalFlowMatching(FlowMatchingConfig(**overrides))


def test_logit_normal_is_the_default_and_concentrates_mid_trajectory() -> None:
    torch.manual_seed(0)

    t = _flow().sample_t(200_000, torch.device("cpu"))

    assert FlowMatchingConfig().t_sampling == "logit_normal"
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0
    assert float(t.mean()) == pytest.approx(0.5, abs=0.01)
    middle = float(((t > 0.25) & (t < 0.75)).float().mean())
    assert middle == pytest.approx(math.erf(math.log(3) / math.sqrt(2)), abs=0.01)


def test_logit_normal_mean_shifts_the_distribution() -> None:
    torch.manual_seed(0)

    t = _flow(t_mean=1.0).sample_t(100_000, torch.device("cpu"))

    assert float(t.median()) == pytest.approx(1 / (1 + math.exp(-1.0)), abs=0.01)


def test_uniform_sampling_remains_available() -> None:
    torch.manual_seed(0)

    t = _flow(t_sampling="uniform").sample_t(200_000, torch.device("cpu"))

    assert float(((t > 0.25) & (t < 0.75)).float().mean()) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("coefficient", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_sway_keeps_the_endpoints_and_stays_increasing(coefficient: float) -> None:
    ts = ConditionalFlowMatching.sway(torch.linspace(0, 1, 101), coefficient)

    assert float(ts[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(ts[-1]) == pytest.approx(1.0, abs=1e-6)
    assert (ts.diff() >= 0).all()


def test_negative_sway_spends_more_steps_near_noise() -> None:
    swayed = ConditionalFlowMatching.sway(torch.linspace(0, 1, 11), -1.0)

    assert float(swayed[1]) < 0.1
    assert float(swayed.diff()[0]) < float(swayed.diff()[-1])


def test_sampling_follows_the_velocity_along_the_swayed_schedule() -> None:
    flow = _flow()
    shape = (3, 4)

    constant = flow.sample(lambda x, t: torch.ones_like(x), shape, steps=7, device=torch.device("cpu"),
                           generator=torch.Generator().manual_seed(0), sway_coefficient=-1.0)
    start = torch.randn(shape, generator=torch.Generator().manual_seed(0))

    assert torch.allclose(constant, start + 1.0, atol=1e-6)
