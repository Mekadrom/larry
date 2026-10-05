import numpy as np
import torch
from torchcodec.decoders import AudioDecoder

from larry.common.utils.convert_np import make_np


def mel_to_db(mels: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    mels = make_np(mels)
    if mels.ndim == 3:
        mels = mels.squeeze(0)
    return mels * 20.0 / np.log(10.0)


def bytes_to_waveforms(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    return AudioDecoder(audio_bytes, sample_rate=sample_rate, num_channels=1).get_all_samples().data
