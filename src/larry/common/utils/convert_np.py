"""This module converts objects into numpy arrays. Copied from torch.utils.tensorboard._convert_np."""

import numpy as np

import torch


def make_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Ensures that an input is an ``np.ndarray`` instance."""
    if isinstance(x, np.ndarray):
        return x

    if np.isscalar(x):
        return np.array([x])

    if isinstance(x, torch.Tensor):
        if x.device.type == "meta":
            return np.random.randn(1)

        return _prepare_pytorch(x)


def _prepare_pytorch(x: torch.Tensor) -> np.ndarray:
    """Casts a pytorch tensor to the correct datatypes and moves off of gpu prior to numpy casting."""
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float16)

    return x.detach().cpu().numpy()
