import torch
from torch import nn


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.zeros_(self.scale_shift[1].weight)
        nn.init.zeros_(self.scale_shift[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.modulation(cond).unsqueeze(1).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


def get_norm_type(type_name: str) -> type[nn.Module]:
    match type_name:
        case "rms":
            return nn.RMSNorm
        case "layer":
            return nn.LayerNorm
        case "ada":
            return AdaLayerNorm
        case _:
            raise ValueError(f"Unknown norm type with type_name={type_name}")
