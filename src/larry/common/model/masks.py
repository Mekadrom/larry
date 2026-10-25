import torch


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Helper to generate a mask tensor to mask positions for lengths smaller than the largest length provided."""
    batch_size = lengths.size(0)
    if max_len <= 0:
        max_len = int(lengths.max().item())

    seq_range = torch.arange(
        0,
        max_len,
        dtype=torch.int64,
        device=lengths.device
    )

    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask
