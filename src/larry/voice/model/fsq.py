from collections.abc import Sequence

import torch
from torch import nn


def fsq_radix(levels: Sequence[int]) -> torch.Tensor:
    return torch.cumprod(torch.tensor([1, *levels[:-1]]), dim=0)


def fsq_digits(unit_ids: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
    radix = fsq_radix(levels).to(unit_ids.device)
    return (unit_ids[..., None] // radix) % torch.tensor(levels, device=unit_ids.device)


def fsq_codes(unit_ids: torch.Tensor, levels: Sequence[int]) -> torch.Tensor:
    half_width = torch.tensor(levels, device=unit_ids.device) // 2
    return (fsq_digits(unit_ids, levels) - half_width) / half_width


class FSQ(nn.Module):
    """Finite Scalar Quantization module.

    This module defines a latent space with a number of allowed "levels" per dimension, any of which may be even or
    odd. This produces a quantized latent space with the product of the levels as its total possible values.
    """

    half_level: torch.Tensor
    offset: torch.Tensor
    shift: torch.Tensor
    half_width: torch.Tensor
    radix: torch.Tensor

    def __init__(self, d_model: int, levels: Sequence[int], eps: float = 1e-3) -> None:
        """Module init that creates an up and down project to make the quantizing bottleneck."""
        super().__init__()
        self.levels = list(levels)
        self.down = nn.Linear(d_model, len(levels))
        self.up = nn.Linear(len(levels), d_model)

        level_tensor = torch.tensor(self.levels, dtype=torch.float32)
        half_level = (level_tensor - 1) * (1 + eps) / 2
        offset = (level_tensor % 2 == 0).float() * 0.5
        self.register_buffer("half_level", half_level, persistent=False)
        self.register_buffer("offset", offset, persistent=False)
        self.register_buffer("shift", torch.atanh(offset / half_level), persistent=False)
        self.register_buffer("half_width", torch.div(level_tensor, 2, rounding_mode="floor"), persistent=False)
        self.register_buffer("radix", fsq_radix(self.levels), persistent=False)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimation hack."""
        z = torch.tanh(self.down(x) + self.shift) * self.half_level - self.offset
        return z + (torch.round(z) - z).detach()

    def indices(self, q: torch.Tensor) -> torch.Tensor:
        """Returns the "vocab" indices represented by the quantized latent vector bottleneck value."""
        return ((q.round() + self.half_width).long() * self.radix).sum(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns both the up-casted post-bottleneck logits and the chosen vocab indices."""
        q = self.quantize(x)
        return self.up(q / self.half_width), self.indices(q)


def code_counts(
        unit_ids: torch.Tensor,
        frame_lengths: torch.Tensor,
        codebook_size: int,
        group_ids: torch.Tensor | None = None,
        n_groups: int = 1,
) -> torch.Tensor:
    """[n_groups, codebook_size] counts of the codes on real frames; padded frames are not counted."""
    frames = torch.arange(unit_ids.shape[1], device=unit_ids.device)
    real = frames.unsqueeze(0) < frame_lengths.to(unit_ids.device).unsqueeze(1)
    if group_ids is None:
        group_ids = torch.zeros(unit_ids.shape[0], dtype=torch.long, device=unit_ids.device)

    groups = group_ids.to(unit_ids.device).unsqueeze(1).expand_as(unit_ids)
    flat = groups[real] * codebook_size + unit_ids[real]
    return torch.bincount(flat, minlength=n_groups * codebook_size).view(n_groups, codebook_size)


def codebook_statistics(counts: torch.Tensor, levels: Sequence[int]) -> dict[str, float]:
    """Usage, perplexity, and per-dimension level shares of one [codebook_size] count vector."""
    total = counts.sum()
    if total == 0:
        return {}

    probs = counts.double() / total
    used = probs[probs > 0]
    statistics = {
        "usage": float((counts > 0).double().mean()),
        "perplexity": float(torch.exp(-(used * used.log()).sum())),
    }

    digits = fsq_digits(torch.arange(len(counts), device=counts.device), levels)
    for d, level_count in enumerate(levels):
        for level in range(level_count):
            share = float(probs[digits[:, d] == level].sum())
            statistics[f"dim{d}/level{level - level_count // 2}"] = share

    return statistics
