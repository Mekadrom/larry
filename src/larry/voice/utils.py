import numpy as np
import torch

from larry.utils.convert_np import make_np


def mel_to_db(mels: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    mels = make_np(mels)
    if mels.ndim == 3:
        mels = mels.squeeze(0)
    return mels * 20.0 / np.log(10.0)
