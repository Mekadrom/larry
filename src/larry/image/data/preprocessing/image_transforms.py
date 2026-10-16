import torch
from torch import nn
from torchvision.transforms.v2 import functional as TVF


class PadToEqualSize(nn.Module):
    """A custom torchvision transform that intelligently pads on all 4 sides as required."""

    def __init__(self, fill_value: float = 0.0, padding_mode: str = "constant") -> None:
        """Configures the padding transform to run on images."""
        super().__init__()
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Runs the padding transform on an input image tensor, padding on all sides as necessary."""
        h, w = TVF.get_size(image)
        max_side = max(h, w)
        left, top = (max_side - w) // 2, (max_side - h) // 2
        return TVF.pad(
            image,
            [left, top, max_side - w - left, max_side - h - top],
            fill=self.fill_value,
            padding_mode=self.padding_mode
        )
