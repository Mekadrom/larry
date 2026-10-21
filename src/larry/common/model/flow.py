import dataclasses
import itertools
import math
from collections.abc import Callable

import torch
from torch import nn

from larry.common.model.model import LarryModelOutput


@dataclasses.dataclass(kw_only=True)
class LarryFlowMatchingOutput(LarryModelOutput):
    targets: torch.Tensor
    mask: torch.Tensor


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        angles = 1000 * t[:, None] * freqs[None]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))


class ConditionalFlowMatching:
    def __init__(self, sigma_min: float) -> None:
        self.sigma_min = sigma_min

    def path(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.rand(x1.shape[0], device=x1.device)
        x0 = torch.randn_like(x1)
        tb = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - (1 - self.sigma_min) * tb) * x0 + tb * x1
        return xt, t, x1 - (1 - self.sigma_min) * x0

    @torch.inference_mode()
    def sample(
            self,
            velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            shape: tuple[int, ...],
            steps: int,
            device: torch.device,
            generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(0, 1, steps + 1, device=device)
        for t0, t1 in itertools.pairwise(ts):
            x = x + (t1 - t0) * velocity_fn(x, t0.expand(shape[0]))

        return x
