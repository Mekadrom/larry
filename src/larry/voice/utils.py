import numpy as np
import torch


def mel_to_db(mels: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(mels, torch.Tensor):
        mels = mels.detach().cpu().numpy()
    if mels.ndim == 3:
        mels = mels.squeeze(0)
    return mels * 20.0 / np.log(10.0)
