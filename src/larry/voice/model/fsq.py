import torch
from torch import nn


class FSQ(nn.Module):
    def __init__(self, d_model: int, dims: int = 8, levels: int = 3) -> None:
        super().__init__()
        assert levels % 2 == 1  # odd levels: symmetric around 0, no half-step offset needed
        self.halve = levels // 2
        self.down = nn.Linear(d_model, dims)
        self.up = nn.Linear(dims, d_model)
        self.register_buffer("radix", levels ** torch.arange(dims))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.down(x)) * self.halve
        return z + (torch.round(z) - z).detach()

    def indices(self, q: torch.Tensor) -> torch.Tensor:
        return ((q.round().long() + self.halve) * self.radix).sum(-1)  # pyright: ignore[reportOperatorIssue]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.quantize(x)
        return self.up(q), self.indices(q)