import torch
from torch import nn


class FSQ(nn.Module):
    """Finite Scalar Quantization module.

    This module defines a latent space with a number of allowed "levels" within a number of dimensions. This produces a
    quantized latent space with levels**dims total possible values.
    """

    def __init__(self, d_model: int, dims: int = 8, levels: int = 3) -> None:
        """Module init that creates an up and down project to make the quantizing bottleneck."""
        super().__init__()
        assert levels % 2 == 1  # odd levels: symmetric around 0, no half-step offset needed
        self.halve = levels // 2
        self.down = nn.Linear(d_model, dims)
        self.up = nn.Linear(dims, d_model)
        self.register_buffer("radix", levels ** torch.arange(dims))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimation hack."""
        z = torch.tanh(self.down(x)) * self.halve
        return z + (torch.round(z) - z).detach()

    def indices(self, q: torch.Tensor) -> torch.Tensor:
        """Returns the "vocab" indices represented by the quantized latent vector bottleneck value."""
        return ((q.round().long() + self.halve) * self.radix).sum(-1)  # pyright: ignore[reportOperatorIssue]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns both the up-casted post-bottleneck logits and the chosen vocab indices."""
        q = self.quantize(x)
        return self.up(q), self.indices(q)


def code_counts(
        token_ids: torch.Tensor,
        frame_lengths: torch.Tensor,
        codebook_size: int,
        group_ids: torch.Tensor | None = None,
        n_groups: int = 1,
) -> torch.Tensor:
    """[n_groups, codebook_size] counts of the codes on real frames; padded frames are not counted."""
    frames = torch.arange(token_ids.shape[1], device=token_ids.device)
    real = frames.unsqueeze(0) < frame_lengths.to(token_ids.device).unsqueeze(1)
    if group_ids is None:
        group_ids = torch.zeros(token_ids.shape[0], dtype=torch.long, device=token_ids.device)

    groups = group_ids.to(token_ids.device).unsqueeze(1).expand_as(token_ids)
    flat = groups[real] * codebook_size + token_ids[real]
    return torch.bincount(flat, minlength=n_groups * codebook_size).view(n_groups, codebook_size)


def codebook_statistics(counts: torch.Tensor, dims: int, levels: int) -> dict[str, float]:
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

    codes = torch.arange(levels ** dims, device=counts.device)
    for d in range(dims):
        digit = (codes // levels ** d) % levels
        for level in range(levels):
            share = float(probs[digit == level].sum())
            statistics[f"dim{d}/level{level - levels // 2}"] = share

    return statistics
