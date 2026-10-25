import torch
from torch import nn


class Patchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        batch_size, channels, height, width = image.shape
        patches = image.view(batch_size, channels, height // p, p, width // p, p)
        # (B, rows, cols, C, p, p) -> (B, rows * cols, C * p * p), row-major
        return patches.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, (height // p) * (width // p), channels * p * p)


class Unpatchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, patches: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
        p = self.patch_size
        batch_size = patches.shape[0]
        channels = patches.shape[-1] // (p * p)
        image = patches.view(batch_size, rows, columns, channels, p, p)
        return image.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, rows * p, columns * p)
