import torch
from torch import nn


class Patchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # todo
        ...


class Unpatchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # todo
        ...
